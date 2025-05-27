# -*- coding: utf-8 -*-
"""
Last modified on May 27, 2025

@author: Hermann Zeyen <hermann.zeyen@gmail.com>
         Université Paris-Saclay

Contains the following functions:
    - plot_geography : Plots geography information is it exists
    - plot_lineaments : Plots lineaments usually measured on tilt angle map
    - histo_plot : Plot data histogram data and allow for interactive choice of
      clipping
    - median_plot : Plots line medians as function of line direction and sensor
    - spectrum_plot : Plots logratihmic power spectrum and linear fits,
      allowing interactive modification of low frequency depths
    - spector1D_plot : Plots depths calculated from spectrum decay of all lines
    - spector2D_plot : Plots map with depths calculated from sliding window
      spectrum decay
    - plot_gradients : Plot first and second vertical as well as horizontal
      gradient maps into three subplots of one figure
    - plot_tilt : Plot tilt angle and tilt angle gradient.
    - ana_histo_plot : Plot cumulated histogram of analytic signal data
    - plot_analytic : Plot map of 2D analytic signal and depth inversion
      results
    - base_plot : Plot diurnal variation data

"""

import numpy as np
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import colors
from ..utilities import transforms as trans
from ..utilities import utilities as utils
from ..in_out.dialog import dialog
from .new_window import newWindow


def plot_geography(ax, geography, dfac):
    """
    Plot geography information into axis "ax".

    Parameters
    ----------
    ax : Matplotlib.Axis object
    geography : dictionary with the following entries:
        - "type" : may be "point" or "line"
        - "x" : x coordinate(s) of point or line
        - "y" : y coordinate(s) of point or line
    dfac : Factor by which to multiply coordinates (usually to pass from m to
        km)

    """
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    # Plot a point
    for key in geography.keys():
        x = np.array(geography[key]["x"]) * dfac
        y = np.array(geography[key]["y"]) * dfac
        # If fpoint outside map don't plot
        if geography[key]["type"] == "POINT":
            if x < xmin or x > xmax or y < ymin or y > ymax:
                continue
            # Check whether a point has already been plotted. If not, add label to legend
            ax.plot(x, y, "o", color="black")
        # Plot a line and check whether a line has already been plotted.
        else:
            ax.plot(x, y, "k")


def plot_lineaments(ax, lineaments, dfac):
    """
    Plot lineaments picked from tilt maps into axis "ax".

    Parameters
    ----------
    ax : Matplotlib.Axis object
    lineaments : dictionary with entries
        - "type" : may be "gravity" or "magnetic"
        - "x" : x coordinate(s) of point or line
        - "y" : y coordinate(s) of point or line
    dfac : float
        Factor by which coordinates are multiplied (usually to pass from m to
            km)
    """
    col = "k"
    for key in lineaments.keys():
        if "gravity" in lineaments[key]["type"]:
            col = "w"
        elif "magnetic" in lineaments[key]["type"]:
            col = "k"
        ax.plot(
            lineaments[key]["x"] * dfac,
            lineaments[key]["y"] * dfac,
            col,
            ls="--",
            linewidth=3,
        )


def histo_plot(data):
    """
    Plot histogram of data and allow for interactive choice of clipping

    Parameters
    ----------
    data : Instance of class DataContainer
        Data to be treated

    Returns
    -------
    min_fix : float
        Minimum accepted value. None if no value clicked
    max_fix : float
        Maximum accepted value. None if no value clicked
    """
    min_fix = None
    max_fix = None
    histo = newWindow("Clip_histogram", 1500, 1000)
    if data.grad_data:
        ax_histo = histo.fig.subplots(1, 2)
    else:
        ax = histo.fig.subplots(1, 1)
        ax_histo = [ax]
    rmin1 = np.nanquantile(data.sensor1, 0.001)
    rmax1 = np.nanmax(data.sensor1)
    counts1, _, _ = ax_histo[0].hist(data.sensor1, 20, (rmin1, rmax1))
    if data.grad_data:
        ax_histo[0].set_title("Sensor 1")
    else:
        if data.data["type"] == "magnetic":
            ax_histo[0].set_title("Magnetic field")
        else:
            ax_histo[0].set_title("Gravity field")
    if data.grad_data:
        rmin2 = np.nanquantile(data.sensor2, 0.001)
        rmax2 = np.nanmax(data.sensor2)
        ax_histo[1].hist(data.sensor2, 20, (rmin2, rmax2))
        ax_histo[1].set_title("Sensor 2")
    histo.setHelp(
        "click within one of the histograms for minimum value, "
        + "then for maximum value. Click outside = None"
    )
    histo.show()
    # Wait for mouse click to define lower data limit
    while True:
        event = histo.get_event()
        if event.name == "button_press_event":
            break
    if event.xdata:
        if data.grad_data:
            if event.xdata > min(data.sensor1.min(), data.sensor2.min()):
                ax_histo[0].plot([event.xdata, event.xdata], [0, max(counts1)], "r")
                min_fix = event.xdata
        else:
            if event.xdata > data.sensor1.min():
                ax_histo[0].plot([event.xdata, event.xdata], [0, max(counts1)], "r")
                min_fix = event.xdata
    # Wait for mouse click to define upper data limit
    while True:
        event = histo.get_event()
        if event.name == "button_press_event":
            break
    if event.xdata:
        if data.grad_data:
            if event.xdata < max(data.sensor1.max(), data.sensor2.max()):
                ax_histo[0].plot([event.xdata, event.xdata], [0, max(counts1)], "r")
                max_fix = event.xdata
        else:
            if event.xdata < data.sensor1.max():
                ax_histo[0].plot([event.xdata, event.xdata], [0, max(counts1)], "r")
                max_fix = event.xdata
    histo.close_window()
    return min_fix, max_fix


def median_plot(
    data,
    median1_even,
    median1_odd,
    median2_even,
    median2_odd,
    nline_even,
    nline_odd,
    txt_even,
    txt_odd,
):
    """
    Plot median values of each line as function of line direction and sensor

    Parameters
    ----------
    data : Instance of class DataContainer
        Data to be treated
    median1_even : Numpy 1D float array
        Contains median values for lines in even direction for sensor 1
    median1_odd :  Numpy 1D float array
        Contains median values for lines in odd direction for sensor 1
    median2_even :  Numpy 1D float array
        Contains median values for lines in even direction for sensor 2
    median2_odd :  Numpy 1D float array
        Contains median values for lines in odd direction for sensor 2
    nline_even :  Numpy 1D int array
        Contains line numbers of even lines
    nline_odd : Numpy 1D int array
        Contains line numbers of odd lines
    txt_even : str
        Text for even lines in legend
    txt_odd : str
        Text for odd lines in legend

    Returns
    -------
    None.

    """
    fig_median = newWindow("Medians", 1500, 1000)
    ax_median = fig_median.fig.add_subplot()
    ax_median.plot(
        nline_even, median1_even, color="b", marker="*", label=f"sensor1 {txt_even}"
    )
    ax_median.plot(
        nline_odd, median1_odd, color="c", marker="*", label=f"sensor1 {txt_odd}"
    )
    if median2_even[0] is not None:
        ax_median.plot(
            nline_even, median2_even, color="r", marker="*", label=f"sensor2 {txt_even}"
        )
        ax_median.plot(
            nline_odd,
            median2_odd,
            color="orange",
            marker="*",
            label=f"sensor2 {txt_odd}",
        )
    ax_median.set_title(f'{data["title"]}: median values of all profiles')
    ax_median.set_xlabel("Number of line")
    if data["type"] == "magnetic":
        ax_median.set_ylabel("Magnetic field [nT]")
    else:
        ax_median.set_ylabel("Gravity field [mGal]")
    ax_median.legend(bbox_to_anchor=(1, 0), loc="lower right")
    fig_median.setHelp("Click ENTER to close")
    fig_median.show()
    while True:
        event = fig_median.get_event()
        if event.name == "key_press_event":
            if event.key == "enter":
                fig_median.close_window()
                break


def spectrum_plot(
    amp,
    k,
    lpos,
    depth1,
    depth2,
    intercept1,
    intercept2,
    max_amps,
    max_pos,
    amp2=None,
    k2=None,
    lpos2=None,
    max_amps2=None,
    max_pos2=None,
):
    """
    Plot spectrum and linear fits and allow interactive modification of
    low frequency depths

    Parameters
    ----------
    amp : numpy 1D float array
        squared amplitudes of spectrum
    k : numpy 1D float array
        wavenumbers
    lpos : float
        Line position (meters)
    depth1 : float
        Depth determined for low frequencies
    depth2 : float
        Depth determined for high frequencies
    intercept1 : float
        Intercept of regression line for low frequencies
    intercept2 : float
        Intercept of regression line for high frequencies
    half_width : int
        Half with to determine local maxima
    max_amps : numpy 1D float array
        Amplitudes at local maxima
    max_pos : numpy 1D float array
        Wave numbers at local maxima
    amp2 : numpy 1D float array, optional; Default: None
        like amp for a possible plot of second spectrum for comparison.
    k2 : numpy 1D float array, optional; Default: None
        like k for a possible plot of second spectrum for comparison.
    lpos2 : float, optional; Default: None
        like lpos for a possible plot of second spectrum for comparison.
    max_amps2 : numpy 1D float array, optional; Default: None
        like max_amps for a possible plot of second spectrum for comparison.
    max_pos2 : numpy 1D float array, optional; Default: None
        like max_pos for a possible plot of second spectrum for comparison.

    Returns
    -------
    depth1 : float
        Depth determined for low frequencies
    intercept1 : float
        Intercept of regression line for low frequencies

    """
    n_Ny = len(k)
    dk = k[1] - k[0]
    xsplit = (intercept2 - intercept1) * 0.5 / (depth2 - depth1)
    isplit = np.argmin(abs(k - xsplit))
    y = np.zeros(n_Ny - 1)
    y[:] = np.nan
    y[: isplit + 1] = intercept1 - k[: isplit + 1] * depth1 * 2.0
    nmx = min(len(k), len(y))
    y[isplit + 2 : nmx] = intercept2 - k[isplit + 2 : nmx] * depth2 * 2.0
    y[isplit + 1] = np.nan
    fig_FFT = newWindow("FFT")
    ax_FFT = fig_FFT.fig.add_subplot()
    ax_FFT.plot(k, amp, "k", label="Spectrum in Y")
    ax_FFT.plot(k[:nmx], y[:nmx], "r", label="Average fit")
    ax_FFT.plot(max_pos, max_amps, "r*")
    xplt1 = np.mean(k[: isplit + 1]) + dk
    yplt1 = np.mean(y[: isplit + 1])
    xplt2 = np.mean(k[isplit + 2 :]) + dk
    yplt2 = np.mean(y[isplit + 2 :])
    ax_FFT.text(
        xplt1,
        yplt1,
        f"{depth1:0.2f}",
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    ax_FFT.text(
        xplt2,
        yplt2,
        f"{depth2:0.2f}",
        verticalalignment="bottom",
        horizontalalignment="left",
    )
    if lpos2 is not None:
        ax_FFT.plot(k2, amp2, "blue", label="Spectrum in X")
        ax_FFT.plot(max_pos2, max_amps2, "g*")
        ax_FFT.set_title(f"Line at {lpos:0.1f}/{lpos2:0.1f}")
        ax_FFT.legend(bbox_to_anchor=(1, 1), loc="upper right", fontsize=10)
    else:
        ax_FFT.set_title(f"Line at {lpos:0.1f}")
    ax_FFT.set_ylabel("ln(Amplitude**2)")
    ax_FFT.set_xlabel("Wavenumber [rad/m]")
    fig_FFT.setHelp(
        "Draw line with left mouse button to get new slope. "
        + "Single click to close without change"
    )
    fig_FFT.show()
    _, coor_x, coor_y = fig_FFT.follow_line(ax_FFT, release_flag=True)
    if len(coor_x) > 0 and coor_x[0] != coor_x[1]:
        depth1 = -((coor_y[1] - coor_y[0]) / (coor_x[1] - coor_x[0])) / 2.0
        intercept1 = coor_y[0] + depth1 * 2 * coor_x[0]
        ax_FFT.text(
            coor_x[0],
            coor_y[0],
            f"  {depth1:0.2f}",
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        fig_FFT.canvas.draw()
        while True:
            event = fig_FFT.get_event()
            if event.name == "button_press_event":
                break
    fig_FFT.close_window()
    return depth1, intercept1


def spector1D_plot(
    data, depths1, intercepts1, depths2, intercepts2, direction, half_width
):
    """
    Plots Spector 1D results and allows modifying depths interactively

    Parameters
    ----------
    data : Instance of class DataContainer
        Data used for spectral analysis
    depths1 : Numpy 1D float array of length number_of_lines
        Calculated depths for low frequencies
    intercepts1 : Numpy 1D float array of length number_of_lines
        Calculated intercept of linear regression line for low frequencies
    depths2 : Numpy 1D float array of length number_of_lines
        Calculated depths for high frequencies
    intercepts2 : umpy 1D float array of length number_of_lines
        Calculated intercept of linear regression line for high frequencies
    direction : int
        If 0, spectra are calculated in Y direction, else in X direction
    half_width : int
        Half width to determine local maxima positions

    Returns
    -------
    depths1 : Numpy 1D float array of length number_of_lines
        Final depths for low frequencies
    intercepts1 : Numpy 1D float array of length number_of_lines
        Final intercept of linear regression line for low frequencies
    depths2 : Numpy 1D float array of length number_of_lines
        Final depths for high frequencies
    intercepts2 : Numpy 1D float array of length number_of_lines
        Final intercept of linear regression line for high frequencies

    """
    if direction:
        dat = data.sensor1_inter.T
        dsamp = data.x_inter[1] - data.x_inter[0]
        lpos = data.y_inter
    else:
        dat = data.sensor1_inter
        dsamp = data.y_inter[1] - data.y_inter[0]
        lpos = data.x_inter
    n_Ny = int(dat.shape[0] / 2)
    while True:
        fig_spector = newWindow("Spector")
        ax_spector = fig_spector.fig.add_subplot()
        ax_spector.plot(lpos, depths1)
        ax_spector.invert_yaxis()
        ax_spector.set_title(
            f'{data.data["title"]}: Average depths from spectral analysis'
        )
        ax_spector.xaxis.set_minor_locator(AutoMinorLocator())
        if direction:
            ax_spector.set_xlabel("Northing of line [m]")
        else:
            ax_spector.set_xlabel("Easting of line [m]")
        ax_spector.set_ylabel("Average depth [m]")
        ax_spector.grid(visible=True, which="both")
        fig_spector.setHelp(
            "Click left mouse button to see spectrum and modify slope;"
            + " right mouse button or ENTER key to finish and close window"
        )
        fig_spector.show()
        while True:
            event = fig_spector.get_event()
            if event.name == "button_press_event":
                break
            elif event.name == "key_press_event":
                if event.key == "enter":
                    break
        if event.name == "key_press_event":
            break
        if event.button != 1:
            break
        if not event.xdata:
            continue
        il = np.argmin(abs(lpos - event.xdata))
        dd, kk = trans.log_spect(data.sensor1_inter[:, il], dsamp, n_Ny)
        max_pos, d, _, _ = utils.min_max(dd, half_width=half_width)
        kkk = kk[max_pos]
        depth1, intercept1 = spectrum_plot(
            dd,
            kk,
            lpos[il],
            depths1[il],
            depths2[il],
            intercepts1[il],
            intercepts2[il],
            d,
            kkk,
        )
        depths1[il] = depth1
        intercepts1[il] = intercept1
        fig_spector.close_window()
    fig_spector.close_window()
    return depths1, intercepts1


def spector2D_plot(
    data,
    depths1,
    intercepts1,
    depths2,
    intercepts2,
    xpos,
    ypos,
    half_width,
    n_Ny,
    window_len,
    nwiny,
    nwinx,
    color,
):
    """
    Plots map of sliding window depth estimations from spectrum decay.
    Allows modifying depths interactively.

    Parameters
    ----------
    data : Instance of class DataContainer
        Data used for spectral analysis
    depths1 : Numpy 1D float array
        Calculated depths for low frequencies
    intercepts1 : Numpy 1D float array
        Calculated intercept of linear regression line for low frequencies
    depths2 : Numpy 1D float array
        Calculated depths for high frequencies
    intercepts2 : umpy 1D float array
        Calculated intercept of linear regression line for high frequencies
    xpos, ypos : Numpy 1D float array
        X and Y coordintes of all calculated depths
    half_width : int
        Half width to determine local maxima positions
    n_Ny : int
        Nyquist number used for depths calculations
    window_len : float
        Length of sliding window in meters
    nwiny, nwinx : int
        Number of calculated points in Y and X directions
    color : matplotlib color scale
        Color scale used for map plot

    Returns
    -------
    depths1 : Numpy 1D float array of length number_of_lines
        Final depths for low frequencies
    intercepts1 : Numpy 1D float array of length number_of_lines
        Final intercept of linear regression line for low frequencies

    """
    x = np.copy(data.x_inter)
    y = np.copy(data.y_inter)
    if y.max() > 100000.0:
        dfac = 0.001
        x *= 0.001
        y *= 0.001
        dunit = "km"
    else:
        dfac = 1.0
        dunit = "m"
    nlin = len(data.lineaments)
    dy = data.y_inter[1] - data.y_inter[0]
    dx = data.x_inter[1] - data.x_inter[0]
    while True:
        fig_spect2 = newWindow("Depths from Spector&Grant")
        ax_spect2 = fig_spect2.plot_images(
            depths1,
            xpos * dfac,
            ypos * dfac,
            ptitle=f"Spectral depth solutions\nwindow length: {window_len} m",
            xlabel=f"Easting [{dunit}]",
            ylabel=f"Northing [{dunit}]",
            clabel="Depth [m]",
            percent=0.005,
            c=color,
        )
        if nlin and data.plotLin_flag:
            plot_lineaments(ax_spect2, data.lineaments, dfac)
        if data.geography_flag:
            plot_geography(ax_spect2, data.geography, dfac)
        ax_spect2.set_xlim([np.nanmin(xpos * dfac), np.nanmax(xpos * dfac)])
        ax_spect2.set_ylim([np.nanmin(ypos * dfac), np.nanmax(ypos * dfac)])
        fig_spect2.setHelp(
            "Left click on map: show results of nearest point; "
            + "Right click on map close window and return to main window"
        )
        fig_spect2.show()
        while True:
            event = fig_spect2.get_event()
            if event.name == "button_press_event":
                break
        if not event.xdata:
            continue
        if event.button > 1:
            fig_spect2.close_window()
            break
        ixclick = np.argmin(abs(xpos * dfac - event.xdata))
        ixpos = np.argmin(abs(data.x_inter - xpos[ixclick]))
        iyclick = np.argmin(abs(ypos * dfac - event.ydata))
        iypos = np.argmin(abs(data.y_inter - ypos[iyclick]))
        data1 = data.sensor1_inter[iypos - nwiny : iypos + nwiny, ixpos]
        dd, kk = trans.log_spect(data1, dy, n_Ny)
        if depths1[iyclick, ixclick]:
            depth1 = depths1[iyclick, ixclick]
            intercept1 = intercepts1[iyclick, ixclick]
            depth2 = depths2[iyclick, ixclick]
            intercept2 = intercepts2[iyclick, ixclick]
        max_pos, d, _, _ = utils.min_max(dd, half_width=half_width)
        kkk = kk[max_pos]
        data2 = data.sensor1_inter[iypos, ixpos - nwinx : ixpos + nwinx]
        dd2, kk2 = trans.log_spect(data2, dx, n_Ny)
        max_pos2, d2, _, _ = utils.min_max(dd2, half_width=half_width)
        kkk2 = kk2[max_pos2]
        depth1, intercept1 = spectrum_plot(
            dd,
            kk,
            xpos[ixclick],
            depth1,
            depth2,
            intercept1,
            intercept2,
            d,
            kkk,
            dd2,
            kk2,
            ypos[iyclick],
            d2,
            kkk2,
        )
        depths1[iyclick, ixclick] = depth1
        intercepts1[iyclick, ixclick] = intercept1
        fig_spect2.close_window()
    fig_spect2.close_window()
    return depths1, intercepts1


def plot_gradients(data_c, color="rainbow"):
    """
    Plot first and second vertical as well as horizontal gradient maps into
    three subplots of one figure

    Parameters
    ----------
    data_c : Instance of class DataContainer
        Data containing the gradients
    color : matplotlib color scale, optional. Default: "rainbow"
        Color scale to be used

    Returns
    -------
    None
    """
    x = data_c.x_inter
    y = data_c.y_inter
    if y.max() > 100000.0:
        dfac = 0.001
        dunit = "km"
    else:
        dfac = 1.0
        dunit = "m"
        nlin = len(data_c.lineaments)
    unit = data_c.unit
    nr, nc = data_c.tilt_ang.shape
    data = data_c.vgrad.reshape(nr, nc, 1)
    data = np.concatenate((data, data_c.vgrad2.reshape(nr, nc, 1)), axis=2)
    data = np.concatenate((data, data_c.hgrad.reshape(nr, nc, 1)), axis=2)
    fig_grad = newWindow("Gradients")
    ax_grad = fig_grad.plot_images(
        data,
        x * dfac,
        y * dfac,
        ptitle=[
            "Vertical gradient",
            "Second vertical derivative",
            "Horizontal gradient",
        ],
        xlabel=[f"Easting [{dunit}]", f"Easting [{dunit}]", f"Easting [{dunit}]"],
        ylabel=[f"Northing [{dunit}]", f"Northing [{dunit}]", f"Northing [{dunit}]"],
        clabel=[
            f"Vert. grad. [{unit}/m]",
            f"2nd vert. grad. [{unit}/m2]",
            f"Hor. grad. [{unit}/m]",
        ],
        c=color,
        percent=0.005,
    )
    if nlin and data_c.plotLin_flag:
        plot_lineaments(ax_grad[0], data_c.lineaments, dfac)
        plot_lineaments(ax_grad[1], data_c.lineaments, dfac)
        plot_lineaments(ax_grad[2], data_c.lineaments, dfac)
    if data_c.geography_flag:
        plot_geography(ax_grad[0], data_c.geography, dfac)
        plot_geography(ax_grad[1], data_c.geography, dfac)
        plot_geography(ax_grad[2], data_c.geography, dfac)
    xmin = np.nanmin(x * dfac)
    xmax = np.nanmax(x * dfac)
    ymin = np.nanmin(y * dfac)
    ymax = np.nanmax(y * dfac)
    ax_grad[0].set_xlim([xmin, xmax])
    ax_grad[0].set_ylim([ymin, ymax])
    ax_grad[1].set_xlim([xmin, xmax])
    ax_grad[1].set_ylim([ymin, ymax])
    ax_grad[2].set_xlim([xmin, xmax])
    ax_grad[2].set_ylim([ymin, ymax])
    fig_grad.show()
    return fig_grad


def plot_tilt(data, color="rainbow"):
    """
    Plot tilt angle and tilt angle gradient. Allows tracing interactively
    lineaments.

    The tilt angle gradients are normalized such that only the gradient maxima
    are shown on the map, independent of their actual value.

    Parameters
    ----------
    data : Instance of class DataContainer
        Data containing the gradients
    color : matplotlib color scale, optional. Default: "rainbow"
        Color scale to be used

    Returns
    -------
    None

    """
    if data.y_inter.max() > 100000.0:
        dfac = 0.001
        dunit = "km"
    else:
        dfac = 1.0
        dunit = "m"
    nlin = len(data.lineaments)
    fig_tilt_ang = newWindow("Tilt angle")
    ax_tilt_ang = fig_tilt_ang.plot_images(
        data.tilt_ang,
        data.x_inter * dfac,
        data.y_inter * dfac,
        ptitle=f"{data.title}\nTilt angle",
        xlabel=f"Easting [{dunit}]",
        ylabel=f"Northing [{dunit}]",
        clabel="Tilt_angle [rad]",
        c=color,
    )
    if data.geography_flag:
        plot_geography(ax_tilt_ang, data.geography, dfac)
    if nlin and data.plotLin_flag:
        plot_lineaments(ax_tilt_ang, data.lineaments, dfac)
    ax_tilt_ang.set_xlim(
        [np.nanmin(data.x_inter * dfac), np.nanmax(data.x_inter * dfac)]
    )
    ax_tilt_ang.set_ylim(
        [np.nanmin(data.y_inter * dfac), np.nanmax(data.y_inter * dfac)]
    )
    fig_tilt_ang.show()

    while True:
        fig_tilt = newWindow("Tilt angle gradient")
        ax_tilt = fig_tilt.plot_images(
            data.tilt_grd,
            data.x_inter * dfac,
            data.y_inter * dfac,
            ptitle=f"{data.title}\nNormalized tilt angle gradient",
            xlabel=f"Easting [{dunit}]",
            ylabel=f"Northing [{dunit}]",
            clabel="Gradient maxima (n.u.)",
            c=color,
        )
        if data.geography_flag:
            plot_geography(ax_tilt, data.geography, dfac)
        if nlin and data.plotLin_flag:
            plot_lineaments(ax_tilt, data.lineaments, dfac)
        ax_tilt.set_xlim(
            [np.nanmin(data.x_inter * dfac), np.nanmax(data.x_inter * dfac)]
        )
        ax_tilt.set_ylim(
            [np.nanmin(data.y_inter * dfac), np.nanmax(data.y_inter * dfac)]
        )
        fig_tilt.setHelp(
            "Left click add point, right click finish line (not integrated). "
            + "Immediate right click without chosen point: stop tilt routine. "
            + "Click with mouse wheel eliminates nearest lineament"
        )
        fig_tilt.show()
        event, coor_x, coor_y = fig_tilt.follow_line(ax_tilt)
        # If follow_line was exited with the wheel button, this means that no new line
        # has been defined. It is then considered that the line nearest to the mouse
        # click should be erased
        if event.button == 2:
            if nlin == 0:
                continue
            if not event.xdata or not event.ydata:
                continue
            xc = event.xdata
            yc = event.ydata
            min_lin = list(data.lineaments.keys())[0]
            min_dist = (xc - data.lineaments[min_lin]["x"][0]) ** 2 + (
                yc - data.lineaments[min_lin]["y"][0]
            ) ** 2
            for ll, lin in data.lineaments.items():
                for i, x in enumerate(lin["x"]):
                    dist = (xc - x) ** 2 + (yc - lin["y"][i]) ** 2
                    if dist < min_dist:
                        min_lin = ll
                        min_dist = dist
            print(f"Erase lineament {min_lin}")
            del data.lineaments[min_lin]
            fig_tilt.close_window()
            continue
        if len(coor_x) == 0:
            fig_tilt.close_window()
            fig_tilt_ang.close_window()
            return
        if len(coor_x) == 1:
            continue
        nlin += 1
        data.lineaments[nlin] = {}
        data.lineaments[nlin]["x"] = np.array(coor_x[:-1]) / dfac
        data.lineaments[nlin]["y"] = np.array(coor_y[:-1]) / dfac
        data.lineaments[nlin]["type"] = data.data["type"]
        data.plotLin_flag = True
        fig_tilt.close_window()


def ana_histo_plot(edges, cum, title):
    """
    Plot cumulative histogram of analytic signal values of one line.
    Allows choosing interactively a clipping value

    Parameters
    ----------
    edges : numpy 1D float array
        Contains the edges of the histogram subdivisions
    cum : numpy 1D float array
        Contains the values of the cumulated histogram in every subdivision
    title : str
        General title of histogram plot

    Returns
    -------
    None
    """
    fig_q = newWindow("Analytic signal quantiles", 1200, 800)
    ax_q = fig_q.fig.add_subplot()
    ax_q.plot(edges, cum)
    ax_q.set_title(f"{title}: Cumulative sum analytic signal")
    ax_q.set_xlabel("value")
    ax_q.set_ylabel("quantile")
    fig_q.setHelp("Click left mouse button to define threshold")
    fig_q.show()
    while True:
        event = fig_q.get_event()
        if event.name == "button_press_event":
            break
    fig_q.close_window()
    return event.xdata


def plot_analytic(data, color="rainbow"):
    """
    Plots 2D analytic signal of sensor 1 and inverts the signal for source
    depths.
    The user is asked to define interactively a clipping value below which
    the calculated analytic signal is ignored. For this, one line is chosen
    on the screen for which the cumulated histogram is plotted for clip-value
    definition. For this line, also the data and inversion result are plotted.
    The calculation may be done on lines in X or Y direction.

    Parameters
    ----------
    data : Instance of class DataContainer
        Data containing the gradients
    color : matplotlib color scale, optional. Default: "rainbow"
        Color scale to be used

    Returns
    -------
    None

    """
    xc = np.copy(data.x_inter)
    yc = np.copy(data.y_inter)
    dbar = None
    gna = None
    dmin = None
    if yc.max() > 100000.0:
        dfac = 0.001
        dunit = "km"
        xc *= dfac
        yc *= dfac
    else:
        dfac = 1.0
        dunit = "m"
    nlin = len(data.lineaments)
    fig_ana = newWindow("Analytic signal")
    ax_ana = fig_ana.plot_images(
        data.analytic_signal1,
        xc,
        yc,
        ptitle="Analytic signal",
        c=color,
        xlabel=f"Easting [{dunit}]",
        ylabel=f"Northing [{dunit}]",
        clabel=f"Analytic signal [{data.unit}/m]",
        percent=0.005,
        bar_height=0.4,
    )
    # If lineaments have been measured and plotting is activated, plot them now
    if nlin and data.plotLin_flag:
        plot_lineaments(ax_ana, data.lineaments, dfac)
    ax_ana.set_xlim([xc.min(), xc.max()])
    ax_ana.set_ylim([yc.min(), yc.max()])
    ax_ana.grid(visible=True, which="both")
    ax_ana.xaxis.set_minor_locator(AutoMinorLocator())
    ax_ana.yaxis.set_minor_locator(AutoMinorLocator())
    fig_ana.setHelp(
        "Click left mouse button to see analytic signal of one "
        + "line in Y, right: in X; Wheel or press ENTER to finish "
        + "and close window"
    )
    fig_ana.show()
    # Interpretation is based on the squared analytic signal
    d2 = data.analytic_signal1**2
    index = np.isnan(d2)
    d2[index] = 0.0
    # Wait for mouse click. Left mouse click will trigger inversion of analytic
    #      signal for depths along N-S lines, right mouse click along E-W lines.
    #      Clicking on wheel finishes this module.
    # The line neares to the mouse click will be shown in an own window with
    #     inversion results
    while True:
        while True:
            event = fig_ana.get_event()
            if event.name == "button_press_event":
                break
            if event.name == "key_press_event":
                if event.key == "enter":
                    break
        if event.name == "key_press_event":
            if event.key == "enter":
                fig_ana.close_window()
                return
        if not event.xdata or not event.ydata:
            continue
        if event.button == 2:
            fig_ana.close_window()
            return
        if event.button == 1:
            nline = np.argmin(abs(xc - event.xdata))
            pos = yc
            pos_line = xc[nline]
            text = f"{np.round(pos_line, 2)} East"
            text_x = f"Northing [{dunit}]"
            nl = len(xc)
            direct = "y"
        else:
            nline = np.argmin(abs(yc - event.ydata))
            pos = xc
            pos_line = yc[nline]
            text = f"{np.round(pos_line, 2)} North"
            text_x = f"Easting [{dunit}]"
            nl = len(yc)
            direct = "x"
        dx = pos[1] - pos[0]
        slope = []
        intercept = []
        depth = []
        alpha = []
        x_center = []
        y_center = []
        fit = []
        pmax = []
        # Mask valaues smaller than 1/1000 of maximum values by setting them to
        #      negative values
        if dmin is None:
            dmn = d2.max() / 1000.0
            dmx = d2.max()
            hist, edges = np.histogram(
                d2.flatten(), bins=1000, range=(dmn, dmx), density=True
            )
            dedge = edges[1] - edges[0]
            cum = np.cumsum(hist) * dedge
            nlim = int(len(cum) * 0.85)
            dmin = ana_histo_plot(edges[1:-nlim], cum[:-nlim], data.title)
            depth_min = 0.0
            depth_max = dx * 20.0 / dfac
        results, okButton = dialog(
            ["Threshold", "Minimum depth", "Maximum depth", "Half width"],
            ["e", "e", "e", "e"],
            [dmin, depth_min, depth_max, 5],
            "Analytic signal parameters",
        )
        if okButton:
            dmin = float(results[0])
            depth_min = float(results[1])
            depth_max = float(results[2])
            half_width = int(results[3])
        dd = np.copy(d2)
        dd[d2 < dmin] = -1.0

        # prepare window with inversion results of clicked line
        fig_sig = newWindow("Analytic signal inversion", 1200, 800)
        # Start loop over lines
        s_best = 1.0e10
        i_best = 0.0
        x_best = 0.0
        dx = pos[1] - pos[0]
        for ll in range(nl):
            if direct == "y":
                d = np.copy(d2[:, ll])
                d_interest = np.copy(d2[:, ll])
                d_interest[d < dmin] = -1.0
                xline = xc[ll]
            else:
                d = np.copy(d2[ll, :])
                d_interest = np.copy(d2[ll, :])
                d_interest[d < dmin] = -1.0
                yline = yc[ll]
            # If no data larger than the threshold exist on the line, continue with next
            # line
            if d_interest.max() < 0.0:
                continue
            # If line corresponds to clicked one, plot analytic signal
            if ll == nline:
                ax_sig = fig_sig.fig.add_subplot()
                ax_sig.plot(pos, d, "k*", label="Analytic signal")
                ax_sig.plot(
                    pos[d_interest > 0],
                    d_interest[d_interest > 0],
                    "g*",
                    label="Unused above threshold",
                )
                ax_sig.set_title(f"Squared analytic signal at {text}")
                ax_sig.set_xlabel(text_x)
                ax_sig.set_ylabel(f"Amplitude [{data.unit}/m]**2")
            # Find maxim and minima along line
            max_pos, _, min_pos, _ = utils.min_max(d_interest, half_width=half_width)
            lab_flag = False
            # Start loop over all found maxima
            for p in max_pos:
                ff = np.inf
                # Search points belonging to the actual maximum. Limit is found when either
                #    a negative value is found (too small analytic signal to be interpreted) or
                #    when the next relative minimum is found
                for k in range(p, -1, -1):
                    if d_interest[k] < 0.0 or k in min_pos:
                        break
                if d_interest[k] < 0:
                    n1 = k + 1
                else:
                    n1 = k
                for k in range(p, len(d_interest)):
                    if d_interest[k] < 0.0 or k in min_pos:
                        break
                if d_interest[k] < 0:
                    n2 = k
                else:
                    n2 = min(k + 1, len(d_interest))
                # If less than 4 points have been found belonging to the maximum, dont threat
                #    this maximum
                if n2 - n1 < half_width:
                    continue
                # Do inversion
                # For this the inverse of squared analytic signal should have a linear
                #     relation with the squared distance from the maximum.
                #     Since the exact position of the maximum is not know, suppose that it is
                #     located in the range measured maximum +/- 1/2*data spacing
                #     The regression line is calculated using the squared amplitude as weight.
                #     In this way, the high amplitudes have a stonger weight for the inversion
                #     than the small ones.
                #     The obtained slope is the inverse of the parameter alpha, the intercept
                #     corresponds to (depth/alpha)**2

                # Calculate inverse of squared amplitude (add a small value to avoid
                # division by 0)
                y = 1.0 / (d_interest[n1:n2] + d_interest[p] * 1.0e-20)
                # Test positions around measured maximum with a step of dx/10 for best solution
                # x contains the squared distance from maximum
                for x0 in np.arange(pos[p] - dx / 2.0, pos[p] + dx / 2.0, dx / 10):
                    x = ((pos[n1:n2] - x0) / dfac) ** 2
                    # G is the Frechet matrix
                    G = np.ones((len(x), 2))
                    G[:, 0] = x
                    # C contains the weights
                    C = np.diag(1.0 / y**2)
                    # Invert for regression line coefficients
                    mat1 = np.matmul(np.transpose(G), C)
                    coefs = np.matmul(
                        np.matmul(np.linalg.inv(np.matmul(mat1, G)), mat1), y
                    )
                    y_fit = np.matmul(G, coefs)
                    f = np.sum((y - y_fit) ** 2)
                    # If fit is better than earlier values, store corresponding parameters, but
                    #    only if slope and intercept are positive
                    if f < ff and coefs[0] > 0.0 and coefs[1] > 0.0:
                        s_best = coefs[0]
                        i_best = coefs[1]
                        x_best = x0
                        ff = f
                # If a best fit was found, store the parameters
                if np.isfinite(ff):
                    dep = np.sqrt(i_best / s_best) - data.h1_sensor
                    if depth_min <= dep <= depth_max:
                        slope.append(s_best)
                        intercept.append(i_best)
                        alpha.append(np.sqrt(1.0 / s_best))
                        depth.append(dep)
                        if direct == "y":
                            y_center.append(x_best)
                            x_center.append(xline)
                        else:
                            y_center.append(yline)
                            x_center.append(x_best)
                        fit.append(np.sqrt(ff))
                        pmax.append(p)
                        # If line corresponds to picked one, plot results into floating window
                        if ll == nline:
                            xx = ((pos[n1:n2] - x_best) / dfac) ** 2
                            yy = xx * s_best + i_best
                            yy = 1.0 / yy
                            if lab_flag:
                                ax_sig.plot(pos[n1:n2], d_interest[n1:n2], "r*")
                                ax_sig.plot(pos[p], d_interest[p], "b*")
                                ax_sig.plot(pos[n1:n2], yy, "r")
                            else:
                                ax_sig.plot(
                                    pos[n1:n2],
                                    d_interest[n1:n2],
                                    "r*",
                                    label="Fitted points",
                                )
                                ax_sig.plot(
                                    pos[p], d_interest[p], "b*", label="Fitted maximum"
                                )
                                ax_sig.plot(
                                    pos[n1:n2], yy, "r", label="Calculated signal"
                                )
                            lab_flag = True
                            ax_sig.text(
                                x_best,
                                yy.max(),
                                f"dep:{depth[-1]:0.1f}, " + f"alfa:{alpha[-1]:0.2f}",
                            )
            if ll == nline:
                ax_sig.legend(bbox_to_anchor=(1, 1), loc="upper right", fontsize=10)
        slope = np.array(slope)
        intercept = np.array(intercept)
        alpha = np.array(alpha)
        depth = np.array(depth)
        x_center = np.array(x_center)
        y_center = np.array(y_center)
        fit = np.array(fit)
        # Store results into file, name depends on the directions used for calculation
        if direct == "y":
            file = "Analytic-signal-solutions_N-S-lines.dat"
        else:
            file = "Analytic-signal-solutions_E-W-lines.dat"
        with open(file, "w", encoding="utf-8") as fo:
            fo.write("     X        Y    Depth[m]   Alpha    Fit\n")
            for i, d in enumerate(depth):
                fo.write(
                    f"{x_center[i]:0.1f} {y_center[i]:0.1f} "
                    + f"{d:0.2f} {alpha[i]:0.3f} {fit[i]:0.6f}\n"
                )
        fig_sig.setHelp("Click any mouse button to close window")
        fig_sig.show()
        # Wait for click within results window to close it
        event = fig_sig.get_event()
        fig_sig.close_window()
        # Plot the obtained depths onto map of analytic signal
        #            vmin = np.quantile(depth,0.01)
        if dbar is not None:
            dbar.remove()
        if gna is not None:
            gna.remove()
        if len(depth) == 0:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "No depths calculated\n\n"
                + f"Probably you placed the threshold too high ({dmin})\n"
                + "\ntry again with larger threshold.",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            continue
        vmin = depth.min()
        vmax = np.quantile(depth, 0.99)
        nc = -int(np.log10(vmax)) + 2
        dv = round((vmax - vmin) / 10, nc)
        vmin = np.ceil(vmin / dv) * dv
        v = list(np.arange(vmin, vmin + 10.5 * dv, dv))
        cmap = plt.get_cmap("hot_r")
        norm = colors.BoundaryNorm(v, cmap.N)
        gna = ax_ana.scatter(x_center, y_center, c=depth, cmap=cmap, norm=norm)
        # Plot colorbar for depths into the axis
        cax = ax_ana.inset_axes([1.015, 0.6, 0.015, 0.35], transform=ax_ana.transAxes)
        dbar = fig_ana.fig.figure.colorbar(
            gna,
            cax=cax,
            extend="max",
            orientation="vertical",
            anchor=(0, 0),
            shrink=0.5,
            aspect=5,
        )
        dbar.ax.tick_params(labelsize=12)
        dbar.ax.set_ylabel("Source depths [m]", fontsize=12)

        ncd = -int(np.log10(dmin)) + 2
        ax_ana.set_title(
            f"Analytic signal and depth solutions in {direct} "
            + f"direction\nthreshold = {np.round(dmin, ncd)}",
            fontsize=14,
        )
        fig_ana.setHelp(
            "Click left mouse button to see analytic signal of one "
            + "line in Y, right: in X; Wheel or ENTER to finish and "
            + "close window"
        )
        fig_ana.canvas.draw()


def base_plot(base, data_type="magnetic"):
    """
    Plot base station data
    Allows defining interactively using the right mouse button areas where to
    mute data (data are set to nan)

    Parameters
    ----------
    base : Instance of class Geometrics
        Contains base station data
    data_type : str
        Usually "magnetics", may also be "gravity"

    Returns
    -------
    None
    """
    while True:
        fig_base = newWindow("Diurnal variations", 800, 500)
        ax_base = fig_base.fig.add_subplot()
        ax_base.plot(base.time_base / 86400.0, base.base)
        ax_base.set_title("Diurnal variations")
        ax_base.set_xlabel("Time of acquisition [day in year]")
        if data_type == "magnetic":
            ax_base.set_ylabel("Magnetic field [nT]")
        else:
            ax_base.set_ylabel("Gravity field [mGal]")
        fig_base.setHelp(
            "Click left mouse button to close, wheel or right "
            + "button at two points to cut out data"
        )
        fig_base.show()
        event = fig_base.get_event()
        if event.name == "key_press_event":
            if event.key == "enter":
                break
        else:
            if event.button == 1:
                break
        x0 = event.xdata * 86400.0
        while True:
            event = fig_base.get_event()
            if event.name == "button_press_event":
                break
        x1 = event.xdata * 86400.0
        index = np.where(
            (base.time_base > min(x0, x1)) & (base.time_base < max(x0, x1))
        )
        base.base[index] = np.nan
        fig_base.close_window()
    fig_base.close_window()
    return base
