# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 17:59:44 2024

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Université Paris-Saclay, France

Contains the following classes for plotting purposes of program PyMaGra:
    plot
    newWindow
"""

import os
import numpy as np
from PyQt5.uic import loadUiType
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator
from matplotlib.path import Path as P
from matplotlib import tri

Ui_MainWindow, QMainWindow = loadUiType(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "Magnetics.ui")
)


class plot(QMainWindow, Ui_MainWindow):
    """
    Class for plotting in PyMaGra

    Depends on the following packages:
        numpy
        PyQt5
        matplotlib

    Needed private files:

        Magnetics.ui (user interface, produced with QtDesigner)

    Contains the following functions:
        __init__
        setClipPath
        set_geography_flag
        toggle_geography_flag
        plot_geography
        plot_lineaments
        plot_triang
        plot_image
        addMPL
        rmMPL
        plotFloating

    """

    def __init__(self, main):
        super().__init__()
        self.setupUi(
            self
        )  # Set up main window based on file Magnetics.ui created with QT Designer
        self.fig = Figure()  # create a first figure in central widget
        self.addMPL(self.fig)
        self.main = main
        self.grid_flag = True
        self.plotLin_flag = False
        self.geography_flag = False
        self.legend = []
        self.grad_data = []
        self.unit = "nT"
        self.bar_or = "vertical"
        self.nticks = 10
        self.axfont = 20

    def setClipPath(self, x, y):
        """
        define clip path

        Parameters
        ----------
        x : numpy float array
            contains all x-coordinates of data points.
        y : numpy float array
            contains all y-coordinates of data points.

        Returns
        -------
        PATH matplotlib.path

        """
        clip_x = []
        clip_y = []
        if self.main.direction == 0:
            for xx in np.unique(x):
                yy = np.round(np.nanmax(y[x == xx]), 2)
                if len(clip_x) == 0:
                    yy0 = yy
                else:
                    if yy != yy0:
                        clip_x.append(xx - 0.01)
                        clip_y.append(yy0)
                clip_x.append(xx)
                clip_y.append(yy)
                yy0 = yy
            for xx in np.unique(x)[::-1]:
                yy = np.round(np.nanmin(y[x == xx]), 2)
                if len(clip_x) == 0:
                    yy0 = yy
                else:
                    if yy != yy0:
                        clip_x.append(xx + 0.01)
                        clip_y.append(yy0)
                clip_x.append(xx)
                clip_y.append(yy)
                yy0 = yy
        else:
            for yy in np.unique(y):
                xx = np.round(np.nanmax(x[y == yy]), 2)
                if len(clip_x) == 0:
                    xx0 = xx
                else:
                    if xx != xx0:
                        clip_y.append(yy - 0.01)
                        clip_x.append(xx0)
                clip_y.append(yy)
                clip_x.append(xx)
                xx0 = xx
            for yy in np.unique(y)[::-1]:
                xx = np.round(np.nanmin(x[y == yy]), 2)
                if len(clip_x) == 0:
                    xx0 = xx
                else:
                    if xx != xx0:
                        clip_y.append(yy + 0.01)
                        clip_x.append(xx0)
                clip_y.append(yy)
                clip_x.append(xx)
                xx0 = xx
        clip_x.append(clip_x[0])
        clip_y.append(clip_y[0])
        clip = np.zeros((len(clip_x), 2))
        clip[:, 0] = np.array(clip_x)
        clip[:, 1] = np.array(clip_y)
        # clip contains the coordinates of the clipping path.
        # codes will contain the way to connect clipping points (move to the first point
        #    of the path, draw lines to all other points and finally close the path)
        codes = []
        Path = P
        codes += [Path.MOVETO]
        codes += [Path.LINETO] * (len(clip_x) - 2)
        codes += [Path.CLOSEPOLY]
        # set clipping
        return Path(clip, codes)

    def set_geography_flag(self, option):
        """
        Set flag to plot geography information or not to value "option"

        Parameters
        ----------
        option : bool
            If True, geography information is plotted.

        """
        self.geography_flag = option

    def toggle_geography_flag(self):
        """
        Inverse value of  flag to plot geography information or not

        No input of output parameters

        """
        self.geography_flag = not self.geography_flag

    def plot_geography(self, ax):
        """
        Plot geography information into axis "ax".

        Parameters
        ----------
        ax : Matplotlib.Axis object

        Geography information is stored in main as dictionary with entries
            "type" : may be "point" or "line"
            "x" : x coordinate(s) of point or line
            "y" : y coordinate(s) of point or line
        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        for key in self.main.geography.keys():
            # Plot a point
            if self.main.geography[key]["type"] == "POINT":
                x = self.main.geography[key]["x"]
                y = self.main.geography[key]["y"]
                # If fpoint outside map don't plot
                if x < xmin or x > xmax or y < ymin or y > ymax:
                    continue
                # Check whether a point has already been plotted. If not, add label to legend
                if "town" in self.legend:
                    ax.plot(x, y, "o", color="black")
                else:
                    self.legend.append("town")
                    ax.plot(x, y, "o", color="black", label="town/place")
                ax.text(
                    x,
                    y,
                    "  " + self.main.geography[key]["name"],
                    ha="left",
                    va="center",
                )
            else:
                # Plot a line and check whether a line has already been plotted.
                if "geology" in self.legend:
                    ax.plot(
                        self.main.geography[key]["x"],
                        self.main.geography[key]["y"],
                        "k",
                    )
                else:
                    self.legend.append("geology")
                    ax.plot(
                        self.main.geography[key]["x"],
                        self.main.geography[key]["y"],
                        "k",
                        label="geology",
                    )

    def plot_lineaments(self, ax):
        """
        Plot lineaments picked from tilt maps into axis "ax".

        Parameters
        ----------
        ax : Matplotlib.Axis object

        Lineament information is stored as dictionary with entries
            "type" : may be "gravity tilt" or "magnetic tilt"
            "x" : x coordinate(s) of point or line
            "y" : y coordinate(s) of point or line
        """
        for key in self.lineaments.keys():
            if self.lineaments[key]["type"] == "gravity tilt":
                col = "w"
            elif self.lineaments[key]["type"] == "magnetic tilt":
                col = "k"
            if self.lineaments[key]["type"] in self.legend:
                ax.plot(
                    self.lineaments[key]["x"],
                    self.lineaments[key]["y"],
                    col,
                    ls="--",
                    linewidth=4,
                )
            else:
                self.legend.append(self.lineaments[key]["type"])
                ax.plot(
                    self.lineaments[key]["x"],
                    self.lineaments[key]["y"],
                    col,
                    label=self.lineaments[key]["type"],
                    ls="--",
                    linewidth=4,
                )

    def plot_north(self, ax, angle, pos):
        """
        Plot North arrow into map
        It is supposed that the map is drawn with set_aspect="equal"

        Parameters
        ----------
        ax : matplotlib axes object
            axis where to plot the North arrow.
        angle : float
            Angle in degrees from N to E (geological sense, not mathematical).
        pos : str
            string of two characters indicationg the position of the arrow.
            One character may be "t" or "b" for top or bottom of the plot
            The other letter may be "l" or "r" for left or right
            The order of the two characters as well as capitalization has no
            importance.

        Returns
        -------
        None.

        """
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = (xmax - xmin) * 0.05
        dy = (ymax - ymin) * 0.05
        alpha = 90.0 - angle
        if alpha < -180:
            alpha += 360.0
        if alpha > 180:
            alpha -= 360.0
        c = np.cos(np.deg2rad(alpha))
        s = np.sin(np.deg2rad(alpha))
        dxa = dy * c
        dya = dy * s
        radius = np.sqrt(dxa**2 + dya**2)

        if "t" in pos.lower():
            y0 = ymax - (1.5 + s) * dy
        else:
            y0 = ymin + (1.5 - s) * dy
        if "r" in pos.lower():
            x0 = xmax - (2.5 + 2 * c) * dx
        else:
            x0 = xmin + (2.5 - 2 * c) * dx

        ax.arrow(
            x0,
            y0,
            dxa,
            dya,
            width=dya / 5.0,
            length_includes_head=True,
            head_length=radius,
            head_width=radius / 1.5,
            shape="left",
            color="k",
            edgecolor=None,
        )
        ax.arrow(
            x0,
            y0,
            dxa,
            dya,
            width=dya / 5.0,
            length_includes_head=True,
            head_length=radius,
            head_width=radius / 1.5,
            shape="right",
            facecolor="w",
            edgecolor="k",
        )
        ax.text(
            x0 + dxa * 1.6,
            y0 + dya * 1.75,
            "N",
            horizontalalignment="center",
            verticalalignment="center_baseline",
            rotation=alpha - 90.0,
            rotation_mode="anchor",
        )

    def plot_sensor_triang(
        self,
        ax,
        x,
        y,
        s,
        vmin,
        vmax,
        percent,
        cmap,
        xlabel,
        ylabel,
        title,
        cbar_title,
        dec=0.0,
    ):
        """
        Plot map of one sensor by triangulation

        Parameters
        ----------
        ax : Matplotlib axis
            Axis where to plot the map
        x : Numpy 1D float array
            X-coordinates of data
        y : Numpy 1D float array
            Y-coordinates of data
        s : Numpy 1D float array
            Values [nT or mGal]
        vmin : float
            Value for lower end of color scale [nT or mGal]
            If vmin==vmax, the color scale is calculated automatically based on
            quantiles (see "percent")
        vmax : float
            Value for upper end of color scale [nT or mGal]
        percent : float
            Scaling of color scale. Maximum value is percentile (1-percent),
            minimum value for percentile percent. This is only used if mincol==maxcol.
        cmap : Matplotlib color map
            Color map to be used
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        title : str
            Title of map
        cbar_title : str
            Text for color bar
        dec : float (optional, default : 0.)
            Direction of the measurement lines with respect to magnetic North
            in degrees, measured from N to E

        Returns
        -------
        None.

        """
        # Define clip paths for both sensors, searching for every line maximum and
        #    minimum coordinates
        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()
        clip_path = self.setClipPath(x, y)
        rd = (
            int(np.ceil(-np.log10(np.nanquantile(s, 0.999) - np.nanquantile(s, 0.001))))
            + 2
        )
        # Define collor bar limits and levels
        if vmin == vmax:
            if percent > 0:
                max_col = np.round(np.nanquantile(s, 1 - percent), rd)
                min_col = np.round(np.nanquantile(s, percent), rd)
            else:
                max_col = np.round(np.nanmax(s), rd)
                min_col = np.round(np.nanmin(s), rd)
        else:
            max_col = np.round(vmax, rd)
            min_col = np.round(vmin, rd)
        rd = int(np.ceil(-np.log10(max_col - min_col))) + 2
        nlev = 51
        ds = np.round((max_col - min_col) / nlev, rd)
        levels = np.arange(min_col, max_col + ds / 2, ds)
        # Calculate triangulation
        triang = tri.Triangulation(x, y)
        # Do triangulation plot
        gci = ax.tricontourf(
            triang,
            s,
            extend="both",
            vmin=min_col,
            vmax=max_col,
            levels=levels,
            cmap=cmap,
        )
        ax.set_clip_on(True)
        for collection in gci.collections:
            collection.set_clip_path(clip_path, transform=ax.transData)
        ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=self.axfont)
        ax.set_ylabel(ylabel, fontsize=self.axfont)
        ax.tick_params(labelsize=self.axfont)
        if self.grid_flag:
            ax.grid(visible=True, which="both")
        if self.plotLin_flag:
            self.plot_lineaments(ax)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if self.bar_or == "vertical":
            cax = ax.inset_axes([1.05, 0.05, 0.02, 0.9], transform=ax.transAxes)
        else:
            cax = ax.inset_axes([0.05, -0.15, 0.9, 0.05], transform=ax.transAxes)
        smin = np.round(np.nanmin(s), rd)
        smax = np.round(np.nanmax(s), rd)
        ssmin = levels[0]
        ssmax = levels[-1]
        ds = (ssmax - ssmin) / self.nticks
        ticks = np.round(np.arange(ssmin, ssmax + ds / 2, ds), rd)
        ticks = list(ticks)
        # Plot color bar
        cbar = plt.colorbar(
            gci,
            orientation=self.bar_or,
            cax=cax,
            fraction=0.1,
            ticks=ticks,
            extend="both",
        )
        if self.bar_or == "vertical":
            cbar.ax.set_ylabel(cbar_title)
            cbar.ax.text(
                0.0,
                -0.06,
                f"{smin}",
                verticalalignment="top",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                0.0,
                1.06,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        else:
            cbar.ax.set_xlabel(cbar_title)
            cbar.ax.text(
                0.0,
                1.0,
                f"{smin}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                1.0,
                1.0,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        ax.set_aspect("equal", adjustable="box")
        if self.geography_flag:
            self.plot_geography(ax)
        if self.plotLin_flag:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper right")
        self.plot_north(ax, -dec, "tl")

    def plot_triang(
        self,
        data,
        title="Measured magnetic data",
        xlabel="Easting",
        ylabel="Northing",
        percent=0.01,
        c="rainbow",
        mincol1=0.0,
        maxcol1=0.0,
        mincol2=0.0,
        maxcol2=0.0,
        mincolg=0.0,
        maxcolg=0.0,
        grad_flag=False,
    ):
        """
        Plot maps of not interpolated data (stn format) of both sensors using
        tricontourf

        Parameters
        ----------
        data: dictionary with one entry per line, key = number of line.
              Each entry is itself a dictionary containing the following entries:
              "s1": Numpy float array with data of sensor 1
              "s2": Numpy float array with data of sensor 2
                    If only data of one single sensor were measured,
                    "s2" contains only one zero.
              "x":  Numpy float array with E-W coordinates of data points
              "y":  Numpy float array with N-S coordinates of data points
              "grad_flag" bool. True if 2 sensors were used, False if only one sensor
              "mask": bool, True if line should be plotted, False if excluded from plotting
        title : str, optional
            Plot title for both sensor polts (the text "sensor n" is automatically
            added, n being sensor numnber). The default is "Measured magnetic data".
        xlabel : str, optional
            Text for x-axis. The default is "Easting".
        ylabel : str, optional
            Text for y-axis. The default is "Northing".
        percent : float, optional
            Scaling of color scale. Maximum value is percentile (1-percent),
            minimum value for percentile percent. The default is 0.01.
            This is only used if mincol==maxcol.
        mincol1 : float, optional
            value for lower end of color scale for sensor 1 [nT]
        maxcol1 : float, optional
            value for upper end of color scale for sensor 1 [nT]
            If mincol1==maxcol1, the color scale is calculated automatically based on
            quantiles (see "percent")
        mincol2 : float, optional
            value for lower end of color scale for sensor 2 [nT]
        maxcol2 : float, optional
            value for upper end of color scale for sensor 2 [nT]
            If mincol2==maxcol2, the color scale is calculated automatically based on
            quantiles (see "percent")
        mincolg : float, optional
            value for lower end of color scale for vertical gradient [nT/m]
        maxcolg : float, optional
            value for upper end of color scale for vertical gradient [nT/m]
            If mincolg==maxcolg, the color scale is calculated automatically based on
            quantiles (see "percent")
        c : str, optional
            Color map to be used. The default is "rainbow".
        grad_flag: bool, optional, default: False
            if True, 3 plots are shown on the screen: Sensor1, sensor2 and gradient.
            if False only the two sensors are plotted

        Returns
        -------
        fig : matplotlib.figure
            Figure used for plot.
        ax : matplotlib.axis
            List of axes used for plots. ax has two entrances, one for each sensor

        """
        x = []
        y = []
        v1 = []
        v2 = []
        self.legend = []
        self.grad_data = data["grad_data"]
        if data["type"] == "magnetic":
            self.unit = "nT"
        else:
            self.unit = "mGal"
        for key, val in data.items():
            try:
                _ = int(key)
                if not val["mask"]:
                    continue
                x += list(val["x"])
                y += list(val["y"])
                v1 += list(val["s1"])
                if self.grad_data:
                    v2 += list(val["s2"])
            except ValueError:
                break
        x = np.array(x)
        y = np.array(y)
        v1 = np.array(v1)
        v2 = np.array(v2)
        ddx = x.max() - x.min()
        ddy = y.max() - y.min()

        if self.grad_data:
            if grad_flag:
                facx = 10 / (3 * ddx)
                facy = 8 / (3 * ddy)
                if facx < facy:
                    fig, ax = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
                    self.bar_or = "vertical"
                    self.nticks = 10
                else:
                    fig, ax = plt.subplots(1, 3, figsize=(20, 8), layout="constrained")
                    self.bar_or = "horizontal"
                    self.nticks = 6
            else:
                facx = 10 / (3 * ddx)
                facy = 8 / (3 * ddy)
                if facx < facy:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
                    self.bar_or = "vertical"
                    self.nticks = 10
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 8), layout="constrained")
                    self.bar_or = "horizontal"
                    self.nticks = 6
        else:
            fig, axis = plt.subplots(1, 1, figsize=(10, 8), layout="constrained")
            ax = [axis]
            self.bar_or = "vertical"
            self.nticks = 10
        matplotlib.rcParams.update({"font.size": 20})
        self.axfont = 20
        plt.rc("xtick", labelsize=self.axfont)
        plt.rc("ytick", labelsize=self.axfont)

        # Eliminate possible nans (coming from function clean_data) for both sensors
        x1 = np.copy(x)
        y1 = np.copy(y)
        s1 = np.copy(v1)
        if self.grad_data:
            x2 = np.copy(x)
            y2 = np.copy(y)
            s2 = np.copy(v2)
            x3 = np.copy(x)
            y3 = np.copy(y)
            s3 = s2 - s1

            index = np.isfinite(s1) & np.isfinite(s2)
            x3 = x3[index]
            y3 = y3[index]
            s3 = s3[index]
            x2 = np.delete(x2, np.isnan(s2))
            y2 = np.delete(y2, np.isnan(s2))
            s2 = np.delete(s2, np.isnan(s2))
        x1 = np.delete(x1, np.isnan(s1))
        y1 = np.delete(y1, np.isnan(s1))
        s1 = np.delete(s1, np.isnan(s1))

        # Do plot for sensor 1
        if self.grad_data:
            if ";" in title[:-1]:
                t = title.split(";")
                txt = t[0] + " Sensor 1;" + t[1]
            else:
                txt = title + " Sensor 1"
        else:
            txt = title
        self.plot_sensor_triang(
            ax[0],
            x1,
            y1,
            s1,
            mincol1,
            maxcol1,
            percent,
            c,
            xlabel,
            ylabel,
            txt,
            f"Field strength [{self.unit}]",
            dec=data["line_declination"],
        )
        if self.grad_data:
            # Do plot for sensor 2
            if ";" in title[:-1]:
                t = title.split(";")
                txt = t[0] + " Sensor 2;" + t[1]
            else:
                txt = title + " Sensor 2"
            self.plot_sensor_triang(
                ax[1],
                x2,
                y2,
                s2,
                mincol2,
                maxcol2,
                percent,
                c,
                xlabel,
                ylabel,
                txt,
                f"Field strength [{self.unit}]",
                dec=data["line_declination"],
            )
            # plot vertical gradient
            if grad_flag:
                if ";" in title[:-1]:
                    t = title.split(";")
                    txt = t[0] + " Gradient;" + t[1]
                else:
                    txt = title + " Gradient"
                self.plot_sensor_triang(
                    ax[2],
                    x3,
                    y3,
                    s3,
                    mincolg,
                    maxcolg,
                    percent,
                    c,
                    xlabel,
                    ylabel,
                    txt,
                    f"Field gradient [{self.unit}/m]",
                    dec=data["line_declination"],
                )
        # Erase actual plot in central widget and plot the new one
        self.rmMPL()
        self.addMPL(fig)
        return fig, ax

    def plot_sensor_image(
        self,
        ax,
        x,
        y,
        s,
        vmin,
        vmax,
        percent,
        cmap,
        xlabel,
        ylabel,
        title,
        cbar_title,
        dec=0.0,
    ):
        """
        Plot map of one sensor by triangulation

        Parameters
        ----------
        ax : Matplotlib axis
            Axis where to plot the map
        x : Numpy 1D float array
            X-coordinates of data columns
        y : Numpy 1D float array
            Y-coordinates of data rows
        s : Numpy 2D float array
            Values [nT or mGal] on a regular grid
        vmin : float
            Value for lower end of color scale [nT or mGal]
            If vmin==vmax, the color scale is calculated automatically based on
            quantiles (see "percent")
        vmax : float
            Value for upper end of color scale [nT or mGal]
        percent : float
            Scaling of color scale. Maximum value is percentile (1-percent),
            minimum value for percentile percent. This is only used if mincol==maxcol.
        cmap : Matplotlib color map
            Color map to be used
        xlabel : str
            Label for x-axis
        ylabel : str
            Label for y-axis
        title : str
            Title of map
        cbar_title : str
            Text for color bar
        dec : float (optional, default : 0.)
            Direction of the measurement lines with respect to magnetic North
            in degrees, measured from N to E

        Returns
        -------
        None.

        """
        matplotlib.rcParams.update({"font.size": 20})
        axfont = 20
        matplotlib.rcParams.update({"font.size": 20})
        plt.rc("xtick", labelsize=axfont)
        plt.rc("ytick", labelsize=axfont)
        # Calculate color scale
        rd = int(np.ceil(-np.log10(np.nanmax(s) - np.nanmin(s)))) + 2
        if vmin == vmax:
            if percent > 0:
                max_col = np.round(np.nanquantile(s, 1 - percent), rd)
                min_col = np.round(np.nanquantile(s, percent), rd)
            else:
                max_col = np.round(np.nanmax(s), rd)
                min_col = np.round(np.nanmin(s), rd)
        else:
            max_col = np.round(vmax, rd)
            min_col = np.round(vmin, rd)
        rd = int(np.ceil(-np.log10(max_col - min_col))) + 2
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        dx2 = (x[1] - x[0]) / 2
        dy2 = (y[1] - y[0]) / 2
        im1 = ax.imshow(
            np.flip(s, axis=0),
            cmap=cmap,
            aspect="equal",
            extent=[xmin - dx2, xmax + dx2, ymin - dy2, ymax + dy2],
            vmin=min_col,
            vmax=max_col,
        )
        if self.grad_data:
            ax.set_title(title)
        else:
            ax.set_title(title)
        ax.set_xlabel(xlabel, fontsize=axfont)
        ax.set_ylabel(ylabel, fontsize=axfont)
        ax.tick_params(labelsize=axfont)
        if self.grid_flag:
            ax.grid(visible=True, which="both")
        if self.plotLin_flag:
            self.plot_lineaments(ax)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        if self.bar_or == "vertical":
            cax = ax.inset_axes([1.05, 0.05, 0.02, 0.9], transform=ax.transAxes)
        else:
            cax = ax.inset_axes([0.05, -0.15, 0.9, 0.05], transform=ax.transAxes)
        smin = np.round(np.nanmin(s), rd)
        smax = np.round(np.nanmax(s), rd)
        ssmin = min_col
        ssmax = max_col
        ds = (ssmax - ssmin) / self.nticks
        ticks = np.round(np.arange(ssmin, ssmax + ds / 2, ds), rd)
        ticks = list(ticks)
        labels = []
        for t in ticks:
            labels.append(f"{t}")
        cbar = plt.colorbar(
            im1,
            orientation=self.bar_or,
            cax=cax,
            fraction=0.1,
            ticks=ticks,
            extend="both",
        )
        if self.bar_or == "vertical":
            #            cbar.ax.set_yticks(ticks,labels=labels)
            cbar.ax.set_ylabel(cbar_title)
            cbar.ax.text(
                0.0,
                -0.06,
                f"{smin}",
                verticalalignment="top",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                0.0,
                1.06,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        else:
            #            cbar.ax.set_xticks(ticks,labels=labels)
            cbar.ax.set_xlabel(cbar_title)
            cbar.ax.text(
                0.0,
                1.0,
                f"{smin}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                1.0,
                1.0,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        if self.geography_flag:
            self.plot_geography(ax)
        #        ax[0].set_aspect('equal', adjustable='box', anchor=anchor)
        ax.set_aspect("equal", adjustable="box")
        if self.plotLin_flag:
            ax.legend(bbox_to_anchor=(1, 1), loc="upper right")
        self.plot_north(ax, -dec, "tl")

    def plot_image(
        self,
        title="Measured magnetic data",
        xlabel="Easting",
        ylabel="Northing",
        percent=0.01,
        c="rainbow",
        mincol1=0,
        maxcol1=0,
        mincol2=0.0,
        maxcol2=0.0,
        mincolg=0.0,
        maxcolg=0.0,
        grad_flag=False,
        dec=0.0,
    ):
        """
        Plot maps of interpolated data of both sensors using imshow

        Parameters
        ----------
        title : str, optional
            Plot title for both sensor polts (the text "sensor n" is automatically
            added, n being sensor numnber). The default is "Measured magnetic data".
        xlabel : str, optional
            Text for x-axis. The default is "Easting".
        ylabel : str, optional
            Text for y-axis. The default is "Northing".
        percent : float, optional
            Scaling of color scale. Maximum value is percentile (1-percent),
            minimum value for percentile percent. The default is 0.01.
        mincol1 : float, optional
            value for lower end of color scale for sensor 1 [nT]
        maxcol1 : float, optional
            value for upper end of color scale for sensor 1 [nT]
            If mincol1==maxcol1, the color scale is calculated automatically based on
            quantiles (see "percent")
        mincol2 : float, optional
            value for lower end of color scale for sensor 2 [nT]
        maxcol2 : float, optional
            value for upper end of color scale for sensor 2 [nT]
            If mincol2==maxcol2, the color scale is calculated automatically based on
            quantiles (see "percent")
        mincolg : float, optional
            value for lower end of color scale for vertical gradient [nT/m]
        maxcolg : float, optional
            value for upper end of color scale for vertical gradient [nT/m]
            If mincolg==maxcolg, the color scale is calculated automatically based on
            quantiles (see "percent")
        c : str, optional
            Color map to be used. The default is "rainbow".
        grad_flag: bool, optional, default: False
            if True, 3 plots are shown on the screen: Sensor1, sensor2 and gradient.
            if False only the two sensors are plotted
        dec : float
            Y direction with respect to geographic north (measured from N to E)

        Returns
        -------
        fig : matplotlib.figure
            Figure used for plot.
        ax : matplotlib.axis
            List of axes used for plots. ax has two entrances, one for each sensor

        """
        self.legend = []
        ddx = self.main.x_inter.max() - self.main.x_inter.min()
        ddy = self.main.y_inter.max() - self.main.y_inter.min()
        if self.grad_data:
            if grad_flag:
                if ddx > 1.5 * ddy:
                    fig, ax = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
                    self.bar_or = "vertical"
                    #                    anchor = 'E'
                    self.nticks = 10
                else:
                    fig, ax = plt.subplots(1, 3, figsize=(20, 8), layout="constrained")
                    self.bar_or = "horizontal"
                    #                    anchor = 'S'
                    self.nticks = 6
            else:
                if ddx > ddy:
                    fig, ax = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
                    self.bar_or = "vertical"
                    #                    anchor = 'E'
                    self.nticks = 10
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 8), layout="constrained")
                    self.bar_or = "horizontal"
                    #                    anchor = 'S'
                    self.nticks = 6
        else:
            fig, axis = plt.subplots(1, 1, figsize=(10, 8), layout="constrained")
            self.bar_or = "vertical"
            #            anchor = 'E'
            ax = [axis]
            self.nticks = 10

        # Do plot for sensor 1
        if self.grad_data:
            if ";" in title[:-1]:
                t = title.split(";")
                txt = t[0] + " Sensor 1;" + t[1]
            else:
                txt = title + " Sensor 1"
        else:
            txt = title
        self.plot_sensor_image(
            ax[0],
            self.main.x_inter,
            self.main.y_inter,
            self.main.sensor1_inter,
            mincol1,
            maxcol1,
            percent,
            c,
            xlabel,
            ylabel,
            txt,
            f"Field strength [{self.unit}]",
            dec=dec,
        )

        # Do plot for sensor 2
        if self.grad_data:
            if ";" in title[:-1]:
                t = title.split(";")
                txt = t[0] + " Sensor 2;" + t[1]
            else:
                txt = title + " Sensor 2"
            self.plot_sensor_image(
                ax[1],
                self.main.x_inter,
                self.main.y_inter,
                self.main.sensor2_inter,
                mincol2,
                maxcol2,
                percent,
                c,
                xlabel,
                ylabel,
                txt,
                f"Field strength [{self.unit}]",
                dec=dec,
            )
            # Do plot for gradient
            if grad_flag:
                if ";" in title[:-1]:
                    t = title.split(";")
                    txt = t[0] + " Gradient;" + t[1]
                else:
                    txt = title + " Gradient"
                self.plot_sensor_image(
                    ax[2],
                    self.main.x_inter,
                    self.main.y_inter,
                    self.main.grad_inter,
                    mincolg,
                    maxcolg,
                    percent,
                    c,
                    xlabel,
                    ylabel,
                    txt,
                    f"Gradient [{self.unit}/m]",
                    dec=dec,
                )
        # Erase actual plot in central widget and plot the new one
        self.rmMPL()
        self.addMPL(fig)
        return fig, ax

    def addMPL(self, fig):
        """
        Add widget to actual Figure
        Input: fig is the actually used Figure (matplotlib.figure.Figure object)
        """
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.mplvl.addWidget(self.toolbar)

    def rmMPL(
        self,
    ):
        """
        Remove widget from actual Figure
        """
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()
        self.mplvl.removeWidget(self.toolbar)
        self.toolbar.close()

    def plotFloating(
        self,
        data,
        x,
        y,
        wtitle="",
        sizeh=800,
        sizev=500,
        mincol=0,
        maxcol=0,
        percent=0,
        c="rainbow",
        ptitle="",
        xlabel="",
        ylabel="",
        clabel="",
    ):
        """
        Plot one or several 2D arrays into floating window

        Parameters
        ----------
        data: numpy array
            data to be plotted
            The array my be 2D or 3D. If 3D, it is supposed that several figures
            should be plotted into the same window in different frames.
            If more than one plot is to be done, the thrid dimension contains
            the different data sets. Every data set must be 2D. Several data sets
            may be concatenated into data with the following commands:
                data = data1.reshape(nr,nc,1)
                data = np.concatenate((data,data2.reshape(nr,nc,1)),axis=2)
                etc.
                data1 and data2 are arrays with shape (nr,nc) defined on regular grids.
        x: numpy 1D array: positions of the columns
        y: numpy 1D array: positions of the rows
        wtitle: str optional, default : empty string
            title of floating window
        sizeh: float optional : horizontal size of floating window; default: 800
        sizev: float optional : vertical size of floating window; default: 500
        mincol: float optional : Minimum value of color scale; default: 0
        maxcol: float optional : Maximum value of color scale; default: 0
                if mincol == maxcol,  and percent == 0, the limits of the color
                scale are minimum and maximum values of the data
        percent: float optional: if >0, mincol and maxcol are calculated as
                quantiles (percent, 1-percent)
        c: str optional : color scale to be used. Default: "rainbow"
        ptitle: str, optional
                may be a single string or a list of strings, where the length
                of the list must correspond to the length of the third
                dimension of data.
                Default: empty string
        xlabel: str, optional
                similar as ptitle for lables of horizontal axis
        ylabel: str, optional
                similar as ptitle for lables of vertical axis
        clabel: str, optional
                similar as ptitle for lables of color bars


        Returns
        -------
        fig_float,ax_float: Matplot values of figure and axes

        """
        # Only 2D arrays may be plotted, test if 1D data are passed to the routine
        if data.ndim < 2:
            _ = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "Function plotFloating is not prepared for 1D plots",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Close,
            )
            return False
        # Create figure
        fig_float = newWindow(wtitle, sizeh, sizev)
        # If 2D array is passed create single axis
        if data.ndim == 2:
            ax_float = fig_float.fig.subplots(1, 1)
            data1 = np.copy(data)
            bar_or = "vertical"
            #            anchor = 'E'
            nticks = 10
        # If 3D array is passed create 2 or 3 axis depending on the shape of data
        else:
            ddx = x.max() - x.min()
            ddy = y.max() - y.min()
            data1 = data[:, :, 0]
            if data.shape[2] == 3:
                # If horizontal extension is > 1.5x vertical one, plot axes vertically one
                #    above the next. If not plot axis in horizontal direction
                facx = sizeh / (3 * ddx)
                facy = sizev / (3 * ddy)
                if facx < facy:
                    ax_float = fig_float.fig.subplots(3, 1)
                    bar_or = "vertical"
                    #                    anchor = 'E'
                    nticks = 10
                else:
                    ax_float = fig_float.fig.subplots(1, 3)
                    bar_or = "horizontal"
                    #                    anchor = 'S'
                    nticks = 6
            else:
                facx = sizeh / (2 * ddx)
                facy = sizev / (2 * ddy)
                if facx < facy:
                    ax_float = fig_float.fig.subplots(2, 1)
                    bar_or = "vertical"
                    #                    anchor = 'E'
                    nticks = 10
                else:
                    ax_float = fig_float.fig.subplots(1, 2)
                    bar_or = "horizontal"
                    #                    anchor = 'S'
                    nticks = 6
        # Calculate color scale for both sensors independently
        rd = int(np.ceil(-np.log10(np.nanmax(abs(data1))))) + 2
        if mincol == maxcol:
            if percent > 0:
                max_col = np.round(np.nanquantile(data1, 1 - percent), rd)
                min_col = np.round(np.nanquantile(data1, percent), rd)
            else:
                max_col = np.nanmax(data1)
                min_col = np.nanmin(data1)
        else:
            max_col = maxcol
            min_col = mincol
        # Plot first into axis ax_float[0]
        try:
            ax = ax_float[0]
            pt = ptitle[0]
            xl = xlabel[0]
            yl = ylabel[0]
            cl = clabel[0]
        # If only one subplot has been created
        except TypeError:
            ax = ax_float
            pt = ptitle
            xl = xlabel
            yl = ylabel
            cl = clabel
        dx2 = (x[1] - x[0]) / 2
        dy2 = (y[1] - y[0]) / 2
        im1 = ax.imshow(
            np.flip(data1, axis=0),
            cmap=c,
            aspect="equal",
            extent=[np.min(x) - dx2, np.max(x) + dx2, np.min(y) - dy2, np.max(y) + dy2],
            vmin=min_col,
            vmax=max_col,
        )
        ax.set_title(pt)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if bar_or == "vertical":
            cax = ax.inset_axes([1.05, 0.05, 0.02, 0.9], transform=ax.transAxes)
        else:
            cax = ax.inset_axes([0.05, -0.15, 0.9, 0.05], transform=ax.transAxes)
        smin = np.round(np.nanmin(data1), rd)
        smax = np.round(np.nanmax(data1), rd)
        ssmin = min_col
        ssmax = max_col
        ds = (ssmax - ssmin) / nticks
        ticks = np.round(np.arange(ssmin, ssmax + ds / 2, ds), rd)
        ticks = list(ticks)
        cbar = plt.colorbar(
            im1, orientation=bar_or, cax=cax, fraction=0.1, extend="both", ticks=ticks
        )
        if bar_or == "vertical":
            cbar.ax.set_ylabel(cl)
            cbar.ax.text(
                0.0,
                -0.075,
                f"{smin}",
                verticalalignment="top",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                0.0,
                1.075,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        else:
            cbar.ax.set_xlabel(cl)
            cbar.ax.text(
                0.0,
                1.0,
                f"{smin}",
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=cbar.ax.transAxes,
            )
            cbar.ax.text(
                1.0,
                1.0,
                f"{smax}",
                verticalalignment="bottom",
                horizontalalignment="left",
                transform=cbar.ax.transAxes,
            )
        if self.grid_flag:
            ax.grid(visible=True)
        #        ax.set_aspect('equal', adjustable='box', anchor=anchor)
        ax.set_aspect("equal", adjustable="box")
        # If more than one map has to be plotted, do this now
        if data.ndim > 2:
            if data.shape[2] > 1:
                data1 = data[:, :, 1]
                rd = int(np.ceil(-np.log10(np.nanmax(abs(data1))))) + 2
                if mincol == maxcol:
                    if percent > 0:
                        max_col = np.round(np.nanquantile(data1, 1 - percent), rd)
                        min_col = np.round(np.nanquantile(data1, percent), rd)
                    else:
                        max_col = np.nanmax(data1)
                        min_col = np.nanmin(data1)
                else:
                    max_col = maxcol
                    min_col = mincol
                im1 = ax_float[1].imshow(
                    np.flip(data1, axis=0),
                    cmap=c,
                    aspect="equal",
                    extent=[
                        np.min(x) - dx2,
                        np.max(x) + dx2,
                        np.min(y) - dy2,
                        np.max(y) + dy2,
                    ],
                    vmin=min_col,
                    vmax=max_col,
                )
                ax_float[1].set_title(ptitle[1])
                ax_float[1].set_xlabel(xlabel[1])
                ax_float[1].set_ylabel(ylabel[1])
                if self.grid_flag:
                    ax_float[1].grid(visible=True)
                if bar_or == "vertical":
                    cax = ax_float[1].inset_axes(
                        [1.05, 0.05, 0.02, 0.9], transform=ax_float[1].transAxes
                    )
                else:
                    cax = ax_float[1].inset_axes(
                        [0.05, -0.15, 0.9, 0.05], transform=ax_float[1].transAxes
                    )
                smin = np.round(np.nanmin(data1), rd)
                smax = np.round(np.nanmax(data1), rd)
                ssmin = min_col
                ssmax = max_col
                ds = (ssmax - ssmin) / nticks
                ticks = np.round(np.arange(ssmin, ssmax + ds / 2, ds), rd)
                ticks = list(ticks)
                cbar = plt.colorbar(
                    im1,
                    orientation=bar_or,
                    cax=cax,
                    fraction=0.1,
                    extend="both",
                    ticks=ticks,
                )
                if bar_or == "vertical":
                    cbar.ax.set_ylabel(clabel[1])
                    cbar.ax.text(
                        0.0,
                        -0.075,
                        f"{smin}",
                        verticalalignment="top",
                        horizontalalignment="left",
                        transform=cbar.ax.transAxes,
                    )
                    cbar.ax.text(
                        0.0,
                        1.075,
                        f"{smax}",
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        transform=cbar.ax.transAxes,
                    )
                else:
                    cbar.ax.set_xlabel(clabel[1])
                    cbar.ax.text(
                        0.0,
                        1.0,
                        f"{smin}",
                        verticalalignment="bottom",
                        horizontalalignment="right",
                        transform=cbar.ax.transAxes,
                    )
                    cbar.ax.text(
                        1.0,
                        1.0,
                        f"{smax}",
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        transform=cbar.ax.transAxes,
                    )
                #                ax_float[1].set_aspect('equal', adjustable='box', anchor=anchor)
                ax_float[1].set_aspect("equal", adjustable="box")
                if data.shape[2] > 2:
                    data1 = data[:, :, 2]
                    rd = int(np.ceil(-np.log10(np.nanmax(abs(data1))))) + 2
                    if mincol == maxcol:
                        if percent > 0:
                            max_col = np.round(np.nanquantile(data1, 1 - percent), rd)
                            min_col = np.round(np.nanquantile(data1, percent), rd)
                        else:
                            max_col = np.nanmax(data1)
                            min_col = np.nanmin(data1)
                    else:
                        max_col = maxcol
                        min_col = mincol
                    im1 = ax_float[2].imshow(
                        np.flip(data1, axis=0),
                        cmap=c,
                        aspect="equal",
                        extent=[
                            np.min(x) - dx2,
                            np.max(x) + dx2,
                            np.min(y) - dy2,
                            np.max(y) + dy2,
                        ],
                        vmin=min_col,
                        vmax=max_col,
                    )
                    ax_float[2].set_title(ptitle[2])
                    ax_float[2].set_xlabel(xlabel[2])
                    ax_float[2].set_ylabel(ylabel[2])
                    if self.grid_flag:
                        ax_float[2].grid(visible=True)
                    if bar_or == "vertical":
                        cax = ax_float[2].inset_axes(
                            [1.05, 0.05, 0.02, 0.9], transform=ax_float[2].transAxes
                        )
                    else:
                        cax = ax_float[1].inset_axes(
                            [0.05, -0.15, 0.9, 0.05], transform=ax_float[1].transAxes
                        )
                    smin = np.round(np.nanmin(data1), rd)
                    smax = np.round(np.nanmax(data1), rd)
                    ssmin = min_col
                    ssmax = max_col
                    ds = (ssmax - ssmin) / nticks
                    ticks = np.round(np.arange(ssmin, ssmax + ds / 2, ds), rd)
                    ticks = list(ticks)
                    cbar = plt.colorbar(
                        im1,
                        orientation=bar_or,
                        cax=cax,
                        fraction=0.1,
                        extend="both",
                        ticks=ticks,
                    )
                    if bar_or == "vertical":
                        cbar.ax.set_ylabel(clabel[2])
                        cbar.ax.text(
                            0.0,
                            -0.075,
                            f"{smin}",
                            verticalalignment="top",
                            horizontalalignment="left",
                            transform=cbar.ax.transAxes,
                        )
                        cbar.ax.text(
                            0.0,
                            1.075,
                            f"{smax}",
                            verticalalignment="bottom",
                            horizontalalignment="left",
                            transform=cbar.ax.transAxes,
                        )
                    else:
                        cbar.ax.set_xlabel(clabel[2])
                        cbar.ax.text(
                            0.0,
                            1.0,
                            f"{smin}",
                            verticalalignment="bottom",
                            horizontalalignment="right",
                            transform=cbar.ax.transAxes,
                        )
                        cbar.ax.text(
                            1.0,
                            1.0,
                            f"{smax}",
                            verticalalignment="bottom",
                            horizontalalignment="left",
                            transform=cbar.ax.transAxes,
                        )
                    #                    ax_float[2].set_aspect('equal', adjustable='box', anchor=anchor)
                    ax_float[2].set_aspect("equal", adjustable="box")
        fig_float.show()
        return fig_float, ax_float


class newWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window.
    """

    def __init__(self, title, xsize=1800, ysize=1200):
        super().__init__()
        self.fig = Figure()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.resize(xsize, ysize)
        self.setWindowTitle(title)
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.canvas.draw()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
