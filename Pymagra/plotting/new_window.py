# -*- coding: utf-8 -*-
"""
Last modified on May 01, 2025

@author: Hermann Zeyen <hermann.zeyen@gmail.com>
         Université Paris-Saclay
"""

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from matplotlib.gridspec import GridSpec


class newWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it will appear as a
    free-floating window.

    contains the following methods:
        - __init__
        - on_enter_event : give focus to window if mouse enters it or moves
            inside
        - close_window : close window and clean up memory
        - setHelp : Write help text at bottom of screen
        - get_event : get keyboard or mouse event with function onClick
        - follow_line : Follow cursor and plots a line from former point
                        with functions onPress, onRelease and onMotion
        - plot_image : Plot a matplotlib imshow image into a subplot
        - plot_images : Plots up to 3 images into 3 subplots
    """

    def __init__(self, title, xsize=1800, ysize=1200, width=20, height=12):
        """
        Initialize a floating window

        Parameters
        ----------
        title : str
            Title of figure window
        xsize, ysize : integers, optional. Defaults: 1800 and 1200
            width and height of figure window in pixels
        width, height : floats, optional. Defaults 15 and 9
            width and height of figure itself in cm
        """
        super().__init__()
        self.fig = plt.figure(figsize=(width, height))
        self.fig.tight_layout()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.resize(xsize, ysize)
        self.setWindowTitle(title)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.cid_enter = self.canvas.mpl_connect("axes_enter_event",
                                                 self.on_enter_event)
        self.canvas.draw()
        self.xsize = xsize
        self.ysize = ysize
        self.width = width
        self.height = height

    def on_enter_event(self, event):
        self.canvas.setFocus()

    # def closeEvent(self, event):
    def close_window(self):
        """Override this method to cleanly close the window."""
# Disconnect matplotlib events
# Explicitly stop any pending drawing operations
# Ensure no pending events are trying to access the canvas
        self.canvas.flush_events()

        if hasattr(self, "cid_enter") and self.cid_enter is not None:
            self.canvas.mpl_disconnect(self.cid_enter)

# Explicitly close the matplotlib figure
        plt.close(self.fig)
# Disconnect the toolbar events (if any)
        if self.toolbar is not None:
            self.toolbar.clear()
# Remove the canvas and toolbar from the layout before deletion
        self.layout.removeWidget(self.canvas)
        self.layout.removeWidget(self.toolbar)

# Ensure that no further actions are performed on the canvas
        self.canvas.setParent(None)  # Detach canvas from the parent widget
        self.toolbar.setParent(None)  # Detach toolbar from the parent widget
# Delete the canvas and figure
        self.canvas.deleteLater()
        self.toolbar.deleteLater()
# Clear the layout (optional but good hygiene)
        while self.layout.count():
            item = self.layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
# Close the window
        self.close()

    def setHelp(self, text):
        """
        Set help text at bottom of screen.

        Parameters
        ----------
        text : str
            Text to be printed (defined in __init__)
            Text is written in QLabel widget. In order to keep the widget at
            the bottom of the screen, the existing one is first erased and
            then reopened. This call has to be done after any modification of
            the graphics window.

        fig : Matplotlib Figure
            Figure where to write the help text.

        Returns
        -------
        None
        """
        try:
            self.help.close()
        except (AttributeError, RuntimeError):
            pass
        self.help = QtWidgets.QLabel(self)
        self.help.setMaximumHeight(20)
        self.layout.addWidget(self.help)
        self.help.show()
        self.help.setText(text)

    def get_event(self):
        """
        wait for a mouse click of keyboard press within figure "fig".

        Parameters
        ----------
        fig : matplotlib figure
            for main window, it must be self.w.mplvl
            for floating window, it is the name that has been given to the
            window at creation time

        Returns
        -------
        event.
        Mouse position, button pressed and the general event parameters are
        accessible through event.xdata, event.ydata, event.button and
        others like event.x or event.y

        """
        self.wait = True

        def onClick(event):
            self.wait = False
            self.event = event

        self.canvas.mpl_connect("button_press_event", onClick)
        self.canvas.mpl_connect("key_press_event", onClick)
        while self.wait:
            QtCore.QCoreApplication.processEvents()
        return self.event

    def follow_line(self, ax, release_flag=False, nleft=1, nright=1):
        """
        Pull line across plot
        Parameters
        ----------
        fig, ax : names of figures and axis where to follow the line
        release_flag (boolean): If True, end of line when left button released\
                                if False, end of line triggered by pressing
                                right button
        nleft (int): if 0 start line for negative direction at origin \
                     if not, start line at the position of first click
        nright (int): if 0 start line for positive direction at origin \
                     if not, start line at the position of first click

        Returns
        -------
        event : QT mouse event
            Information concerning the last mouse event.
            Mainly useful are event.button (mouse button pressed/released);
            event.xdata and event.ydata: coordinates of last mouse position in
            axis coordinate system (e.g. meters);
            event.x and event.y: coordinates of last mouse position in
            window coordinates (pixels)

        coor_x and coor_y: list of floats
            coordinates (axis coordinate system) of line segment(s)

        Mouse button pressed to exit sunction (may be 2 for wheel ar 3
        for right button)

        Left button adds new point. right button finishes. Wheel erases last
        clicked point or, if no points are available, returns
        """
# set flags and initialize coordinates
        self.released = False
        self.start = []
        self.coor_x = []
        self.coor_y = []
        (self.line,) = ax.plot(self.coor_x, self.coor_y, "k", animated=True)

        def onPress(event):
            self.event = event
            self.line_click = False
# left mouse button is pressed
            if event.button == 1:
                if event.xdata is None or event.ydata is None:
                    self.mouse = 1
                    self.x_event = event.xdata
                    self.y_event = event.ydata
                    return
                if len(self.coor_x) == 0:
                    if (event.xdata < 0 and nleft == 0) or\
                            (event.xdata >= 0 and nright == 0):
                        self.start = [0, 0]
                        self.coor_x.append(0)
                        self.coor_y.append(0)
                    else:
                        self.start = [event.xdata, event.ydata]
                        self.coor_x.append(event.xdata)
                        self.coor_y.append(event.ydata)
                    if event.xdata < 0:
                        self.side = -1
                    else:
                        self.side = 1
                    self.background = self.fig.canvas.copy_from_bbox(
                        self.fig.bbox)
# set starting point initially also as end point
                self.coor_x.append(event.xdata)
                self.coor_y.append(event.ydata)
                self.canvas_follow = self.line.figure.canvas
                self.axl = self.line.axes
                self.line.set_data(self.coor_x, self.coor_y)
                self.axl.draw_artist(self.line)
# set action on mouse motion
                self.cidmotion = self.line.figure.canvas.mpl_connect(
                    "motion_notify_event", onMotion)
# set action on mouse release
                if release_flag:
                    self.cidrelease = self.line.figure.canvas.mpl_connect(
                        "button_release_event", onRelease)
# if right button is pressed, finish
            elif event.button == 3:
                self.mouse = 3
                self.x_event = event.xdata
                self.y_event = event.ydata
                try:
                    self.line.figure.canvas.mpl_disconnect(self.cidpress)
                except NameError:
                    pass
                self.line_click = False
                if len(self.coor_x) > 0:
                    self.line_click = True
                self.line.set_animated(False)
                if self.line_click:
                    self.fig.canvas.restore_region(self.background)
                self.background = None
                self.released = True
                return
# Wheel is pressed, erase last point
            else:
                if len(self.coor_x) > 0:
                    print(f"Erase point ({self.coor_x[-1]},{self.coor_y[-1]})")
                    del self.coor_x[-1]
                    del self.coor_y[-1]
                    self.canvas_follow = self.line.figure.canvas
                    self.axl = self.line.axes
                    self.line.set_data(self.coor_x, self.coor_y)
                    self.axl.draw_artist(self.line)
# set action on mouse motion
                    self.cidmotion = self.line.figure.canvas.mpl_connect(
                        "motion_notify_event", onMotion)
                else:
                    self.mouse = 2
                    self.x_event = event.xdata
                    self.y_event = event.ydata
                    try:
                        self.line.figure.canvas.mpl_disconnect(self.cidpress)
                    except (NameError, AttributeError):
                        pass
                    self.line_click = False
                    try:
                        self.line.figure.canvas.mpl_disconnect(self.cidmotion)
                    except (NameError, AttributeError):
                        pass
                    if len(self.coor_x) > 0:
                        self.line_click = True
                    self.line.set_animated(False)
                    if self.line_click:
                        self.fig.canvas.restore_region(self.background)
                    self.background = None
                    self.released = True
                    return

        def onRelease(event):
            """
            When mouse is released, finish line segment

            Parameters
            ----------
            event : QT mouse event
                Not really needed, but necessary for routine call

            Returns
            -------
            None.

            """
# If line finishes when button is released do this here
            global figure
            self.line.figure.canvas.mpl_disconnect(self.cidpress)
            self.line.figure.canvas.mpl_disconnect(self.cidmotion)
            self.line.figure.canvas.mpl_disconnect(self.cidrelease)
            self.line.set_animated(False)
            if len(self.coor_x) > 0:
                self.fig.canvas.restore_region(self.background)
            self.background = None
            self.released = True
            return False

        def onMotion(event):
            """
            When mouse is moved, redraw the line from the fixed starting point
            to the new mouse position

            Parameters
            ----------
            event : QT mouse event
                contains mainly coordinates of new mouse position

            Returns
            -------
            None.

            """
            global figure
            self.event = event
            if event.xdata is None or event.ydata is None:
                return False
# set second point of line as actual mouse position
            self.coor_x[-1] = event.xdata
            self.coor_y[-1] = event.ydata
# Draw new line
            self.line.set_data(self.coor_x, self.coor_y)
            self.line.set_color("k")
            self.canvas_follow = self.line.figure.canvas
            self.axl = self.line.axes
            self.fig.canvas.restore_region(self.background)
            self.axl.draw_artist(self.line)
            self.canvas_follow.blit(self.axl.bbox)
            return True

# set action on mouse motion
        self.cidpress = self.line.figure.canvas.mpl_connect(
            "button_press_event", onPress)
# As long as release flag is not set listen to events
        while self.released is not True:
            QtCore.QCoreApplication.processEvents()
# Return event information and vector of line coordinates
        return self.event, self.coor_x, self.coor_y

    def plot_image(self, ax, data, x, y, mincol, maxcol, percent, c, ptitle,
                   xlabel, ylabel, clabel, grid_flag, fontsize, bar_or,
                   nticks, bar_height):
        """
        Plot one 2D arrays into floating window using matplotlib's imshow.

        Parameters
        ----------
        ax : Matplotlib Axis
            Axis where to plot the image
        data : 2D numpy float array
        x : numpy 1D float array
            Positions of the columns
        y : numpy 1D float array
            Positions of the rows
        mincol : float
            Minimum value of color scale
        maxcol : float
            Maximum value of color scale.
            If mincol == maxcol,  and percent == 0, the limits of the color
            scale are minimum and maximum values of the data.
        percent : float
            If >0, mincol and maxcol are calculated as quantiles
            (percent, 1-percent).
        c: str
            color scale to be used.
        ptitle : str
            Plot title
        xlabel : str
            Lable of horizontal axis
        ylabel : str
            Lable of vertical axis
        clabel : str
            Lable of color bar
        grid_flag : bool
            If True, grid of major ticks is plotted
        fontsize : int
            Fontsize of title. Fontsizes of all other labels or numbers are
            reduced by 2 points
        bar_or : str
            Color bar orientation? May be "horizontal" of "vertical"
        nticks : int
            approximate number of ticks desired for color bar

        Returns
        -------
        ax_float: List of Matplot axes

        """
        rd = int(np.ceil(-np.log10(np.nanmax(abs(data))))) + 2
        if mincol == maxcol:
            if percent > 0:
                max_col = np.round(np.nanquantile(data, 1 - percent), rd)
                min_col = np.round(np.nanquantile(data, percent), rd)
            else:
                max_col = np.nanmax(data)
                min_col = np.nanmin(data)
        else:
            max_col = maxcol
            min_col = mincol
        dx2 = (x[1] - x[0]) / 2
        dy2 = (y[1] - y[0]) / 2
        im1 = ax.imshow(
            np.flip(data, axis=0),
            cmap=c,
            aspect="equal",
            extent=[np.min(x)-dx2, np.max(x)+dx2, np.min(y)-dy2,
                    np.max(y)+dy2], vmin=min_col, vmax=max_col)
        ax.set_title(ptitle, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize - 2)
        ax.set_ylabel(ylabel, fontsize=fontsize - 2)
        ax.tick_params(axis="both", which="major", labelsize=fontsize - 2)
        if bar_or == "vertical":
            cax = ax.inset_axes([1.015, 0.05, 0.03, bar_height],
                                transform=ax.transAxes)
        else:
            cax = ax.inset_axes([0.05, -0.15, bar_height, 0.05],
                                transform=ax.transAxes)
        smin = np.round(np.nanmin(data), rd)
        smax = np.round(np.nanmax(data), rd)
        ssmin = min_col
        ssmax = max_col
        ds = (ssmax - ssmin)/nticks
        ticks = np.round(np.arange(ssmin, ssmax+ds/2, ds), rd)
        ticks = list(ticks)
        cbar = self.fig.colorbar(
            im1, orientation=bar_or, ax=ax, cax=cax, fraction=0.2,
            extend="both", ticks=ticks)
        if bar_or == "vertical":
            cbar.ax.set_ylabel(clabel, size=fontsize-2)
            cbar.ax.text(
                0.0, -0.075, f"{smin}", verticalalignment="top",
                horizontalalignment="left", transform=cbar.ax.transAxes,
                fontsize=fontsize-2)
            cbar.ax.text(
                0.0, 1.075, f"{smax}", verticalalignment="bottom",
                horizontalalignment="left", transform=cbar.ax.transAxes,
                fontsize=fontsize-2)
            for lab in cbar.ax.yaxis.get_ticklabels():
                lab.set_fontsize(fontsize-2)
        else:
            cbar.ax.set_xlabel(clabel, size=fontsize-2)
            cbar.ax.text(
                0.0, 1.0, f"{smin}", verticalalignment="bottom",
                horizontalalignment="right", transform=cbar.ax.transAxes,
                fontsize=fontsize - 2)
            cbar.ax.text(
                1.0, 1.0, f"{smax}", verticalalignment="bottom",
                horizontalalignment="left", transform=cbar.ax.transAxes,
                fontsize=fontsize - 2)
            for lab in cbar.ax.xaxis.get_ticklabels():
                lab.set_fontsize(fontsize - 2)
        if grid_flag:
            ax.grid(visible=True)
        ax.set_aspect("equal", adjustable="box")

    def plot_images(self, data, x, y, mincol=0, maxcol=0, percent=0,
                    c="rainbow", ptitle="", xlabel="", ylabel="", clabel="",
                    grid_flag=True, bar_height=0.9):
        """
        Plot one or several 2D arrays into floating window

        Parameters
        ----------
        data : numpy array
            Data to be plotted.
            The array my be 2D or 3D. If 3D, it is supposed that several
            figures should be plotted into the same window in different frames.
            If more than one plot is to be done, the thrid dimension contains
            the different data sets. Every data set must be 2D. Several data
            sets may be concatenated into data with the following commands:

            - data = data1.reshape(nr,nc,1)
            - data = np.concatenate((data,data2.reshape(nr,nc,1)),axis=2)

            data1 and data2 are arrays with shape (nr,nc) defined on regular
            grids.

        x : numpy 1D float array
            Positions of the columns
        y : numpy 1D float array
            Positions of the rows
        wtitle : str optional, default: empty string
            Title of floating window
        mincol : float optional, default: 0
            Minimum value of color scale
        maxcol : float optional,  default: 0
            Maximum value of color scale.
            If mincol == maxcol,  and percent == 0, the limits of the color
            scale are minimum and maximum values of the data.
        percent : float optional:
            If >0, mincol and maxcol are calculated as quantiles
            (percent, 1-percent).
        c: str optional, default: "rainbow"
            color scale to be used.
        ptitle : str, optional, default: empty string
            May be a single string or a list of strings, where the length of
            the list must correspond to the length of the third dimension of
            data.
        xlabel : str, optional
            Similar as ptitle for lables of horizontal axis
        ylabel : str, optional
            Similar as ptitle for lables of vertical axis
        clabel : str, optional
            Similar as ptitle for lables of color bars

        Returns
        -------
        ax_float: List of Matplot axes

        """
# Only 2D arrays may be plotted, test if 1D data are passed to the routine
        if data.ndim < 2:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "Function plotFloating is not prepared for 1D plots",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return False
# Create figure
        self.fig.tight_layout(w_pad=15, h_pad=2)
# If 2D array is passed create single axis
        ax_float = []
        if data.ndim == 2:
            self.gs = GridSpec(16, 10, self.fig)
            ax_float.append(self.fig.add_subplot(self.gs[1:15, 1:9]))
            data1 = np.copy(data)
            bar_or = "vertical"
            nticks = 10
            fontsz = 14
# If 3D array is passed create 2 or 3 axis depending on the shape of data
        else:
            ddx = x.max() - x.min()
            ddy = y.max() - y.min()
            data1 = data[:, :, 0]
# If horizontal extension is > 1.5x vertical one, plot axes vertically one
#    above the next. If not plot axis in horizontal direction
            if data.shape[2] == 3:
                facx = self.xsize/(3.*ddx)
                facy = self.ysize/(3.*ddy)
                if facx < facy:
                    self.gs = GridSpec(30, 10, self.fig)
                    ax_float.append(self.fig.add_subplot(self.gs[1:8, 1:9]))
                    ax_float.append(self.fig.add_subplot(self.gs[11:18, 1:9]))
                    ax_float.append(self.fig.add_subplot(self.gs[21:28, 1:9]))
                    bar_or = "vertical"
                    nticks = 10
                    fontsz = 10
                else:
                    self.gs = GridSpec(10, 30, self.fig)
                    ax_float.append(self.fig.add_subplot(self.gs[1:9, 1:8]))
                    ax_float.append(self.fig.add_subplot(self.gs[1:9, 11:18]))
                    ax_float.append(self.fig.add_subplot(self.gs[1:9, 21:28]))
                    bar_or = "horizontal"
                    nticks = 6
                    fontsz = 10
            else:
                facx = self.xsize/(2.*ddx)
                facy = self.ysize/(2.*ddy)
                if facx < facy:
                    self.gs = GridSpec(20, 10, self.fig)
                    ax_float.append(self.fig.add_subplot(self.gs[1:8, 1:9]))
                    ax_float.append(self.fig.add_subplot(self.gs[11:18, 1:9]))
                    bar_or = "vertical"
                    nticks = 10
                    fontsz = 12
                else:
                    self.gs = GridSpec(10, 20, self.fig)
                    ax_float.append(self.fig.add_subplot(self.gs[1:9, 1:8]))
                    ax_float.append(self.fig.add_subplot(self.gs[1:9, 11:18]))
                    bar_or = "horizontal"
                    nticks = 6
                    fontsz = 12
# Plot first into axis ax_float[0]
        if isinstance(ptitle, list):
            pt = ptitle[0]
            xl = xlabel[0]
            yl = ylabel[0]
            cl = clabel[0]
        else:
            pt = ptitle
            xl = xlabel
            yl = ylabel
            cl = clabel
        self.plot_image(ax_float[0], data1, x, y, mincol, maxcol, percent, c,
                        pt, xl, yl, cl, grid_flag, fontsz, bar_or, nticks,
                        bar_height)
# If more than one map has to be plotted, do this now
        if data.ndim > 2:
            if data.shape[2] > 1:
                self.plot_image(ax_float[1], data[:, :, 1], x, y, mincol,
                                maxcol, percent, c, ptitle[1], xlabel[1],
                                ylabel[1], clabel[1], grid_flag, fontsz,
                                bar_or, nticks, bar_height)
# If a third plot is to be done, do this now
                if data.shape[2] > 2:
                    self.plot_image(ax_float[2], data[:, :, 2], x, y, mincol,
                                    maxcol, percent, c, ptitle[2], xlabel[2],
                                    ylabel[2], clabel[2], grid_flag, fontsz,
                                    bar_or, nticks, bar_height)
        if len(ax_float) == 1:
            return ax_float[0]
        return ax_float
