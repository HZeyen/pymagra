
# -*- coding: utf-8 -*-
"""
last modified on May 01, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         University Paris-Saclay, France

Needed Python packages:
    PyQt5
    matplotlib
    numpy
    os
    sys
    signal
    sklearn
    scipy

Needed private files:

    Geometrics.py
    Magnetics.ui


Contains the following class:
    Main : Main class, controlling inputs, data treatment and outputs

"""

import os
from copy import deepcopy
from datetime import datetime, date
from signal import signal, SIGINT
import numpy as np
from PyQt5 import QtWidgets
from .in_out import io
from .plotting.main_window import mainWindow
from .plotting import floating_plots as FP
from .utilities import utilities as u
from .in_out.dialog import dialog
from .data.data import DataContainer
from .inversion.inver import inversion
from .inversion.potential_prism import Prism_calc as PP


class Main(QtWidgets.QWidget):
    """
    Main class for PyMaGra Program

    Parameters
    ----------
    dir0 : str (default = None)
        name of working folder

    Methods
    -------
    - __init__ : Initialization
    - readdata : Input data sets
    - check_data : Check if magnetic data contain errors (mainly nulls)
    - correct_time : Correct instrument time
    - file_change : plot another already opened data file
    - join_data : Join all available data sets into one common set.
    - join_gridded : Join all available gridded data sets into a common set.
    - getGeography : Get geography information (towns, geological borders...)
    - oneOfTwo : Extract one line out of two (for lines in alternating
      directions)
    - readBaseData : Input base station data
    - writeStn : Write treated data in Geometrics .stn format
    - writeDat : Write treated data in Geometrics Surfer format
    - writeGXF : Write gridded data in GXF format
    - saveBaseData : Write base station data into Geometrics .str file
    - original : Reset actual data to the original ones
    - plotOriginal : Plot original data set (but keep treated data as actual
      ones)
    - plotActual : Plot actual, treated data to screen
    - plot_gradient : Toggle on/off plotting of gradient data if there are
    - plot_geography : Toggle on/off plotting of geography data
    - plot_lineaments : Toggle on/off plotting of measured lineaments
    - plot_grid : Activate or deactivate plotting of grid lines
    - plot_points : Activate or deactivate plotting of data point positions
    - plot_line : Plot a lineament onto the data map
    - plotLine : Wrapper for plot_Line
    - changeColorScale : Change parameters for Color scale
    - plotBase : Plot base station data
    - plot_median : Plot medians of all lines
    - zooming : (TODO)
    - zoomOut : (TODO)
    - zoomIn : (TODO)
    - zoomInitial : (TODO)
    - save_plot : Save actual plot into .png file
    - diurnal : Estimate diurnal variations from variation of line medians
    - clean : Eliminate outliers from data
    - justify_median : Justify medians of lines measured in opposite directions
    - justify_gauss : Justify gaussian distribution of lines in opposite
      directions
    - interpol : Interpolate data onto a regular grid
    - nan_fill : Interpolate data at positions have nan-value
    - reduce_pole : Reduce magnetic data to the pole
    - spector : Calculate 1D source depths from data spectrum
    - spector2D : Calculate 1D source depths from data spectrum on a 2D grid
    - tilt : Calculate tilt angle of 2D data set
    - continuation : Continue potential field data set upwards
    - analytic_signal : Calculate analytic signal and source depths
    - inver2D : Do 2.5D inversion
    - inver3D : Do 3D inversion
    - synthetic_model : Define synthetic model and calculate its effect
    - Handler : Should handle exeptions - not really working
    - save_lineaments : Saves measured data lineaments into file
    - closeApp : Close application in an ordered way

    """

    def __init__(self, dir0=None):
        super().__init__()

        try:
            os.chdir(dir0)
        except NameError:
            pass
        self.data_read = False
        self.base_files = []
        self.dat = []
        self.dat_ori = []
        self.data_files = []
        self.file_types = []
        self.data_types = []
        self.config_file = []
        self.percent = 0.01
        self.mincol1 = 0.0
        self.maxcol1 = 0.0
        self.mincol2 = 0.0
        self.maxcol2 = 0.0
        self.mincolg = 0.0
        self.maxcolg = 0.0
        self.d_sensor = 0.9
        self.h_sensor = 0.4
        self.height = 1.3
        self.gradient_flag = True
        self.inter_flag = False
        self.config_flag = False
        self.nan_flag = False
        self.geography_flag = False
        self.point_flag = False
        self.n_blocks = 0
        self.grid_flag = True
        self.nlineaments = 0
        self.color = "rainbow"
        self.lineaments = {}
        self.treatments = {}
        self.treatments["diurnal"] = False
        self.treatments["clip"] = False
        self.treatments["justify_median"] = False
        self.treatments["justify_Gauss"] = False
        self.treatments["interpol"] = False
        self.treatments["nan_fill"] = False
        self.treatments["pole"] = False
        self.treatments["odd lines"] = False
        self.treatments["even lines"] = False
        self.treatments["up"] = False
        self.line_choice = "all"
        self.dir0 = dir0
        self.fig_base = None
        self.inclination = 62.0
        self.declination = 0.0
        self.field_flag = False
        self.geography = {}
        self.base = None
        self.fig_line = None
        self.ax_line = None
        self.ax_base = None
        self.fig_median = None
        self.ax_median = None
        self.histo = None
        self.ax_histo = None
        self.fig_spector = None
        self.ax_spector = None
        self.fig_FFT = None
        self.ax_FFT = None
        self.fig_spect2 = None
        self.ax_spect2 = None
        self.fig_FFT2 = None
        self.ax_FFT2 = None
        self.fig_grad = None
        self.ax_grad = None
        self.fig_tilt = None
        self.ax_tilt = None
        self.fig_ana = None
        self.ax_ana = None
        self.fig_q = None
        self.ax_q = None
        self.fig_sig = None
        self.ax_sig = None
        self.wait = True
        self.click = False
        self.press = False
        self.event = None
        self.x_event = None
        self.y_event = None
        self.coor_x = 0.0
        self.coor_y = 0.0
        self.line = None
        self.released = False
        self.line_click = None
        self.x_mouse = 0.0
        self.y_mouse = 0.0
        self.mouse = None
        self.xmin = 0.0
        self.ymin = 0.0
        self.xwin = 0.0
        self.ywin = 0.0
        self.day_joint_flag = False
        self.diff_weight = 1.0
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        self.sensor1_inter = []
        self.sensor2_inter = []
        self.sensor1_back = []
        self.sensor2_back = []
        self.sensor1_fill = []
        self.sensor2_fill = []
        self.grad_inter = []
        self.grad_fill = []
        self.x_inter = []
        self.y_inter = []
        self.z_inter = []
        self.mask1 = []
        self.mask2 = []
        self.start = [0.0, 0.0]
        self.side = 0
        self.background = None
        self.canvas = None
        self.axl = None
        self.cidmotion = None
        self.cidrelease = None
        self.cidpress = None
# string_keys gives the number of keys in data that are not numbers
        self.string_keys = 0
        self.w = mainWindow(self)

# Input data
        self.actual_plotted_file = 0
        self.readData(dir0)
        x = self.dat[0].data[0]["x"]
        y = self.dat[0].data[0]["y"]
        dx = x.max() - x.min()
        dy = y.max() - y.min()
        if dx > dy:
            self.direction = 1
        else:
            self.direction = 0
        if "m" not in self.data_types[0]:
            self.w.save_base.setEnabled(False)
            self.w.readBase.setEnabled(False)
            self.w.basePlot.setEnabled(False)
            self.w.plotGradient.setEnabled(False)
            self.w.diurnalCorrection.setEnabled(False)
            self.w.medianJustify.setEnabled(False)
            self.w.poleReduction.setEnabled(False)

        self.help = QtWidgets.QLabel(self)
        QtWidgets.qApp.installEventFilter(self)

# Define actions for Menu buttons
# Actions for menu File
        self.w.addData.triggered.connect(self.readData)
        self.w.saveSTN.triggered.connect(self.writeStn)
        self.w.saveDat.triggered.connect(self.writeDat)
        self.w.saveGXF.triggered.connect(self.writeGXF)
        self.w.readBase.triggered.connect(self.readBaseData)
        self.w.save_base.triggered.connect(self.saveBaseData)
        self.w.Save_plot.triggered.connect(self.save_plot)
        self.w.geography.triggered.connect(self.getGeography)
        self.w.quitAction.triggered.connect(self.close_app)
# Actions for menu Display
        self.w.originalPlot.triggered.connect(self.plotOriginal)
        self.w.actualPlot.triggered.connect(self.plotActual)
        self.w.change_file.triggered.connect(self.file_change)
        self.w.join.triggered.connect(self.join_data)
        self.w.plotLine.triggered.connect(self.plot_line)
        self.w.basePlot.triggered.connect(self.plotBase)
        self.w.medianPlot.triggered.connect(self.plot_median)
        self.w.plotGradient.triggered.connect(self.plot_gradient)
        self.w.plotGeo.triggered.connect(self.plot_geography)
        self.w.plotGrid.triggered.connect(self.plot_grid)
        self.w.plotLineaments.triggered.connect(self.plot_lineaments)
        self.w.fill.triggered.connect(self.nan_fill)
        self.w.measurement_points.triggered.connect(self.plot_points)
        # self.w.zoom.triggered.connect(self.zooming)
        # self.w.zoom_Out.triggered.connect(self.zoomOut)
        # self.w.zoom_In.triggered.connect(self.zoomIn)
        # self.w.zoom_Initial.triggered.connect(self.zoomInitial)
        self.w.changeQuantile.triggered.connect(self.changeColorScale)
        self.w.secondLine.triggered.connect(self.oneOfTwo)
# Actions for menu Utilities
        self.w.originalData.triggered.connect(self.original)
        self.w.cleanData.triggered.connect(self.clean)
        self.w.timeCorrect.triggered.connect(self.correct_time)
        # self.w.adjust.triggered.connect(self.block_adjust)
        self.w.diurnalCorrection.triggered.connect(self.diurnal)
        self.w.medianJustify.triggered.connect(self.justify_median)
        self.w.gaussJustify.triggered.connect(self.justify_gauss)
        self.w.interpolate.triggered.connect(self.interpol)
        self.w.poleReduction.triggered.connect(self.reduce_pole)
        self.w.lineFFT.triggered.connect(self.spector)
        self.w.Spector_2D.triggered.connect(self.spector2D)
        self.w.tiltAngle.triggered.connect(self.tilt)
        self.w.prolongation.triggered.connect(self.continuation)
        self.w.analytic.triggered.connect(self.analytic_signal)
# Actions for menu Inversion
        self.w.inv2D.triggered.connect(self.inver2D)
        self.w.inv3D.triggered.connect(self.inver3D)
        self.w.synthetic.triggered.connect(self.synthetic_model)

# Check whether file "lineaments.dat" exists containing magnetic or
# gravity lineaments picked from tilt angle maps. If so, create
# dictionary xith all lineament information.
        if os.path.isfile("lineaments.dat"):
            self.lineaments = io.read_lineaments("lineaments.dat")
            self.nlineaments = len(self.lineaments)
            self.w.plotLineaments.setEnabled(True)
            self.w.plotLin_flag = True
            self.w.lineaments = self.lineaments
            for d in self.dat:
                d.lineaments = self.lineaments
                d.lineament_flag = True
                d.plotLin_flag = True
            self.data.lineaments = self.lineaments
            self.data.lineament_flag = True
            self.data.plotLin_flag = True
# Intercept CTL-C to exit in a controlled way
        signal(SIGINT, self.Handler)
        self.w.grad_flag = False
        self.grad_flag = False
        self.gradient_flag = False
        if self.data_types[self.actual_plotted_file] == "magnetic":
            self.unit = "nT"
        else:
            self.unit = "mGal"
# Plot data
        self.fig, self.ax = self.w.plot_triang(
            self.data, title=f"{self.data.data['title']}",
            percent=self.percent, mincol1=self.mincol1, maxcol1=self.maxcol1,
            mincol2=self.mincol2, maxcol2=self.maxcol2, mincolg=self.mincolg,
            maxcolg=self.maxcolg, grad_flag=self.gradient_flag)

    def readData(self, dir0):
        """
        Reads additional data files to the ones read during init of Geometrics
        normally, Geometrics stn files should be read, but another format is
        also possible, considered to be output of program mgwin.
        See Geometrics.read_txt for more information

        Returns
        -------
        None.

        """
        df, ft, dt, self.dir0 = io.get_files(dir0)
        self.data_files += df
        self.file_types += ft
        self.data_types += dt
        ld = len(self.dat)
# Define the first read data file as active one
        self.actual_plotted_file = ld
        for i, f in enumerate(df):
            fconfig = os.path.basename(f)
            j = fconfig.rfind(".")
# Check whether there is a configuration file for each data file
            if j > 0:
                fconfig = fconfig[:j] + ".config"
            else:
                fconfig += ".config"
            self.config_file.append(fconfig)
            tcorr_flag = False
# If there is, read its content
            if os.path.isfile(fconfig):
                self.config_flag = True
                with open(fconfig, "r") as fi:
                    lines = fi.readlines()
                self.file_type = lines[1][:-1]
                self.title_text = lines[2][:-1]
                self.line_dec = float(lines[3])
                self.height_sens1 = -float(lines[4])
                self.height_sens2 = -float(lines[5])
                self.sensor_disposition = int(lines[6][0])
                self.strength = float(lines[7])
                self.inclination = float(lines[8])
                self.declination = float(lines[9])
                if len(lines) > 10:
                    tcorr_flag = True
                    tcorr = float(lines[10])
            else:
                self.config_flag = False
            self.n_blocks += 1

# Create a data class for the actual file
            self.dat.append(DataContainer(self.n_blocks))
            self.dat[-1].set_values(file=df[i], ftype=ft[i], dtype=dt[i])

# Read Geometics (*.stn) file
            if ft[i] == "GEOMETRICS":
                if len(self.dat) > 1:
                    self.dat[-1].set_values(
                        line_declination=self.dat[-2].line_declination,
                        h_sensor=self.dat[-2].h1_sensor,
                        h2_sensor=self.dat[-2].h2_sensor,
                        d_sensor=self.dat[-2].d_sensor,
                        dispo=self.dat[-2].dispo)
                if self.config_flag:
                    self.dat[-1].read_geometrics(
                        f, self.height_sens1, self.height_sens2,
                        self.sensor_disposition, self.line_dec,
                        self.title_text)
                else:
                    self.dat[-1].read_geometrics(f)
                self.w.saveSTN.setEnabled(True)
                self.w.fill.setEnabled(True)
# Read GXF format file
            elif ft[i] == "GXF":
                if self.config_flag:
                    self.dat[-1].read_gxf(
                        f, self.height_sens1, self.line_dec, self.title_text)
                else:
                    self.dat[-1].read_gxf(f)
                self.gradient_flag = False
                self.w.fill.setEnabled(True)
                self.w.plotGradient.setEnabled(False)
# Read BRGM aeromagnetic data
            elif ft[i] == "BRGM":
                self.dat[-1].read_BRGM_flight(f, self.title_text)
                self.gradient_flag = False
                self.w.fill.setEnabled(True)
                self.w.plotGradient.setEnabled(False)
                self.w.poleReduction.setEnabled(False)
# Read *.dat data (usually coming from program MGWIN)
            elif ft[i] == "MGWIN":
                if self.config_flag:
                    self.dat[-1].read_txt(
                        f, self.height_sens1, self.height_sens2, self.line_dec,
                        self.title_text)
                else:
                    self.dat[-1].read_txt(f)
                if self.dat[-1]["grad_data"]:
                    self.gradient_flag = True
                    self.w.plotGradient.setEnabled(True)
                else:
                    self.gradient_flag = False
                    self.w.plotGradient.setEnabled(False)
                self.w.fill.setEnabled(True)
            self.dat[-1].set_values(
                block=self.n_blocks, block_name=f"{os.path.basename(df[i])}",
                typ=self.data_types[i], inter_flag=False,
                treatments=self.treatments, fconfig=fconfig)
            if tcorr_flag:
                self.dat[-1].correct_time(dt=tcorr)
                self.w.timeCorrect.setEnabled(False)
            self.dat_ori.append(deepcopy(self.dat[-1]))
            self.check_data(self.dat[-1].data, f)
# If data are magnetic, no configuration file exits and Earth's field
#   properties were not yet read, ask now for these properties
            line_dir = self.dat[-1].data["line_declination"]
            if "m" in self.data_types[i]:
                if not self.field_flag and not self.config_flag:
                    self.earth = io.get_mag_field(line_dir)
                    self.inclination = self.earth.inc
                    self.declination = self.earth.dec
                    self.strength = self.earth.f
                    self.field_flag = True
# If a configuration file exists but Earth's field properties were not yet
#   defined, create earthMag calss and store the field data.
                elif not self.field_flag:
                    self.earth = io.get_mag_field(
                        line_dir, self.strength, self.inclination,
                        self.declination)
                    self.field_flag = True
            else:
                self.earth = io.get_mag_field(line_dir, 0.0, 0.0, 0.0)
            self.inclination_ori = self.inclination
            self.declination_ori = self.declination
            self.earth_ori = deepcopy(self.earth)
            self.dat[-1].set_values(earth=self.earth)
# If the configuration file did not yet exist, create it now with the data
#   read in via the dialog boxes
            if not self.config_flag:
                with open(fconfig, "w") as fo:
                    fo.write(f"{ft[i]}\n")
                    fo.write(f"{self.data_types[i]}\n")
                    fo.write(f'{self.dat[-1].data["title"]}\n')
                    fo.write(f'{self.dat[-1].data["line_declination"]}\n')
                    fo.write(f'{self.dat[-1].data["height"]}\n')
                    if self.dat[-1].data["grad_data"]:
                        self.height2 = self.dat[-1].data["height2"]
                        self.dist_sensors = self.dat[-1].data["height"] - \
                            self.height2
                    else:
                        self.height2 = self.dat[-1].data["height"]
                        self.dist_sensors = 0.0
                    fo.write(f"{self.height2}\n")
                    fo.write(f'{self.dat[-1].data["dispo"]}\n')
                    fo.write(f"{self.earth.f}\n")
                    fo.write(f"{self.earth.inc}\n")
                    dec = self.earth.dec
                    fo.write(f'{dec+self.dat[-1].data["line_declination"]}\n')
        if len(self.dat) > 1:
            self.w.change_file.setEnabled(True)
            self.w.join.setEnabled(True)
        self.data = deepcopy(self.dat[self.actual_plotted_file])
        self.height = self.data.data["height"]
        self.string_keys = 0
        data_keys = list(self.data.data.keys())
        for i, k in enumerate(data_keys):
            if isinstance(k, (str)):
                break
        self.string_keys = len(data_keys) - i
# Plot data from active file
        if ld > 0:
            self.plotActual()

    def check_data(self, data, file):
        """
        Check magnetic data for consitency

        Data acquired with PPM or Cs/Rb magnetometers may have null measurement
        if the sensors are placed parallel or perpendicular to the Earth's
        field. Program checks whether the minimum or maximum values in a data
        set are smaller/larger than the median by more than 20000 nT and if so,
        it gives a warning message proposing to clean up the data using
        Utilities -> Clean data before any other treatment.

        Parameters
        ----------
        data : dictionary
            Contains all data of a data set. entries "s1", "s2" and "grad_data"
            are used
        file : str
            File name of the data set (only used in the warning message).

        Returns
        -------
        None.

        """
        vmin1 = 1000000.0
        vmax1 = -1000000.0
        vmin2 = 1000000.0
        vmax2 = -1000000.0
        vmed = data[0]["median1"]
        err_flag = False
        grad_test = data["grad_data"]
        for key, val in data.items():
            if isinstance(key, (str)):
                break
            vmin1 = min(val["s1"].min(), vmin1)
            vmax1 = max(val["s1"].max(), vmax1)
            if grad_test:
                vmin2 = min(val["s2"].min(), vmin2)
                vmax2 = max(val["s2"].max(), vmax2)
        if vmax1 > vmed + 20000.0 or vmin1 < vmed - 20000.0:
            err_flag = True
        if grad_test:
            if vmax2 > vmed + 20000.0 or vmin2 < vmed - 20000.0:
                err_flag = True
        if err_flag:
            if grad_test:
                _ = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    f"File {file}:\n\n"
                    + f"Sensor 1: min: {vmin1:0.2f}, max: {vmax1:0.2f}"
                    + f"\nSensor 2: min: {vmin2:0.2f}, max: {vmax2:0.2f}"
                    + "\n\nConsider cleaning up data as first step.",
                    QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            else:
                _ = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    f"File {file}:\n\n"
                    + f"Data: min: {vmin1:0.2f}, max: {vmax1:0.2f}"
                    + "\n\nConsider cleaning up data as first step.",
                    QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)

    def correct_time(self, dt=None):
        """
        Wrapper for measurement time correction

        Parameters
        ----------
        dt : float, optional, Default: None
            Time in seconds to be added to the actual time. If None, dt is
            asked for interactively.

        Returns
        -------
        None.

        """
        self.data.correct_time(dt)

    def file_change(self):
        """
        Choose another file to be represented and treated.

        Dialogue box presents a list of available files other than the one
        actually on the screen and allows to choose by clicking a radio button.
        Clicking on Cancel keeps the actual data.

        Returns
        -------
        None.

        """
        self.dat[self.actual_plotted_file] = deepcopy(self.data)
        labels = [[]]
        file_nr = []
        for i, d in enumerate(self.dat):
            if i == self.actual_plotted_file:
                continue
            labels[0].append(f"{d.data['block_name']}")
            file_nr.append(i)
        if len(file_nr) > 1:
            results, ok_button = dialog(labels, "r", "0",
                                        title="choose data file")
            if not ok_button:
                return None
            self.actual_plotted_file = file_nr[int(results[0])]
        else:
            self.actual_plotted_file = file_nr[0]
        self.data = deepcopy(self.dat[self.actual_plotted_file])
        self.inter_flag = self.data.inter_flag
        if self.inter_flag:
            self.x_inter = np.copy(self.data.x_inter)
            self.y_inter = np.copy(self.data.y_inter)
            self.sensor1_inter = np.copy(self.data.sensor1_inter)
            if self.data.data["grad_data"]:
                self.sensor2_inter = np.copy(self.data.sensor2_inter)
                self.grad_inter = np.copy(self.data.grad_inter)
        self.treatments = deepcopy(self.data.treatments)
        self.plotActual()

    def join_data(self):
        """
        Joins all available data sets into one common data set

        All data must have the same data type (magnetic vs gravity) and be
        located in the same coordinate system. If the different data blocks are
        not contiguous in space, program will interpolate meaningless "data".

        The original data sets are maintained such that using file_change, one
        may return to one of the smaller data sets. If later a new data set is
        added and data are joint again, the joint data set is excluded.

        Returns
        -------
        None.

        """
        self.dat[self.actual_plotted_file] = deepcopy(self.data)
        nbk = len(self.dat)
        self.n_blocks += 1
        self.dat.append(DataContainer(self.n_blocks))
        nlines = 0
        blkn = "blocks"
        self.dat[-1].data = deepcopy(self.dat[0].data)
        for key, val in self.dat[0].data.items():
            if isinstance(key, str):
                del self.dat[-1].data[key]
        pos_lines = []
        blkn = f"{blkn} {self.dat[0].data['block']}"
        for key, val in self.dat[0].data.items():
            if isinstance(key, (str)):
                break
            if val["direction"] in ("N", "S", 0.0, 180.0):
                pos_lines.append(np.median(val["x"]))
            else:
                pos_lines.append(np.median(val["y"]))
        nlines = len(pos_lines)
# Before joining datasets, they must all be treated in the same way except for
# cleaning, which may be necessary for one file but not for another one
        self.treatments = deepcopy(self.dat[0].treatments)
        tr = deepcopy(self.treatments)
        del tr["clip"]
        for i, d in enumerate(self.dat[1:-1]):
            dt = deepcopy(d.treatments)
            del dt["clip"]
            if dt != tr:
                _ = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    f"Data set {d.data['block']}:\n\n"
                    + "Data treatments are not the same as in other blocks\n\n"
                    + "All blocks must have been treated in the same way "
                    + "before joining\n\nJoin data sets is aborted",
                    QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
                self.n_blocks -= 1
                del self.dat[-1]
                return
            if i == nbk:
                break
# Skip already joint data sets (they contain the character "+" in their name)
            if "+" in d.data["block_name"]:
                continue
            blkn = f"{blkn}+{d.data['block']}"
            self.dat[-1].xmin = min(d.xmin, self.dat[-1].xmin)
            self.dat[-1].xmax = max(d.xmax, self.dat[-1].xmax)
            self.dat[-1].ymin = min(d.ymin, self.dat[-1].ymin)
            self.dat[-1].ymax = max(d.ymax, self.dat[-1].ymax)
            for key, val in d.data.items():
                add_flag = False
                if isinstance(key, (str)):
                    break
                if val["direction"] in ("N", "S", 0.0, 180.0):
                    pos_line = np.median(val["x"])
                else:
                    pos_line = np.median(val["y"])
                for il, p in enumerate(pos_lines):
                    if np.isclose(pos_line, p):
                        add_flag = True
                        break
                if add_flag:
                    self.dat[-1].data[il]["x"] = np.concatenate(
                        (self.dat[-1].data[il]["x"], val["x"]))
                    self.dat[-1].data[il]["y"] = np.concatenate(
                        (self.dat[-1].data[il]["y"], val["y"]))
                    if val["direction"] in ("N", "S", 0.0, 180.0):
                        index = np.argsort(self.dat[-1].data[il]["y"])
                    else:
                        index = np.argsort(self.dat[-1].data[il]["x"])
                    self.dat[-1].data[il]["x"] = \
                        self.dat[-1].data[il]["x"][index]
                    self.dat[-1].data[il]["y"] = \
                        self.dat[-1].data[il]["y"][index]
                    self.dat[-1].data[il]["z"] = np.concatenate(
                        (self.dat[-1].data[il]["z"], val["z"]))[index]
                    self.dat[-1].data[il]["s1"] = np.concatenate(
                        (self.dat[-1].data[il]["s1"], val["s1"]))[index]
                    if len(val["s2"]) > 1:
                        self.dat[-1].data[il]["s2"] = np.concatenate(
                            (self.dat[-1].data[il]["s2"], val["s2"]))[index]
                    else:
                        self.dat[-1].data[il]["s2"] = np.concatenate(
                            (self.dat[-1].data[il]["s2"], val["s2"]))
                    self.dat[-1].data[il]["time"] = np.concatenate(
                        (self.dat[-1].data[il]["time"], val["time"]))[index]
                    self.dat[-1].data[il]["topo"] = np.concatenate(
                        (self.dat[-1].data[il]["topo"], val["topo"]))[index]
                    if self.dat[-1].data[il]["direction"] in (
                            "N", "S", 0.0, 180.0):
                        _, index = np.unique(
                            self.dat[-1].data[il]["y"], return_index=True)
                    else:
                        _, index = np.unique(
                            self.dat[-1].data[il]["x"], return_index=True)
                    self.dat[-1].data[il]["x"] = \
                        self.dat[-1].data[il]["x"][index]
                    self.dat[-1].data[il]["y"] = \
                        self.dat[-1].data[il]["y"][index]
                    self.dat[-1].data[il]["z"] = \
                        self.dat[-1].data[il]["z"][index]
                    self.dat[-1].data[il]["s1"] = \
                        self.dat[-1].data[il]["s1"][index]
                    if self.data.data["grad_data"]:
                        self.dat[-1].data[il]["s2"] = \
                            self.dat[-1].data[il]["s2"][index]
                    self.dat[-1].data[il]["time"] = \
                        self.dat[-1].data[il]["time"][index]
                else:
                    self.dat[-1].data[nlines] = deepcopy(val)
                    nlines += 1
                    pos_lines.append(pos_line)

            self.dat[-1].sensor1 = np.concatenate((self.dat[-1].sensor1,
                                                   d.sensor1))
            self.dat[-1].sensor2 = np.concatenate((self.dat[-1].sensor2,
                                                   d.sensor2))
            self.dat[-1].x = np.concatenate((self.dat[-1].x, d.x))
            self.dat[-1].y = np.concatenate((self.dat[-1].y, d.y))
            self.dat[-1].z = np.concatenate((self.dat[-1].z, d.z))
            self.dat[-1].topo = np.concatenate((self.dat[-1].topo, d.topo))
            self.dat[-1].time = np.concatenate((self.dat[-1].time, d.time))
        for key, val in self.dat[0].data.items():
            if isinstance(key, (str)):
                self.dat[-1].data[key] = deepcopy(self.dat[0].data[key])
        self.dat[-1].grad_data = self.dat[0].data["grad_data"]
        self.dat[-1].data["block"] = self.n_blocks
        self.dat[-1].data["block_name"] = blkn
        self.dat[-1].data["title"] += f", {blkn}"
        self.dat[-1].data["type"] = self.data_types[0]
        self.dat[-1].line_declination = self.dat[0].line_declination
        self.dat[-1].h_sensor = self.dat[0].h1_sensor
        self.dat[-1].d_sensor = self.dat[0].d_sensor
        self.dat[-1].dispo = self.dat[0].dispo
        self.dat[-1].data["grad_data"] = self.dat[0].data["grad_data"]
        self.dat[-1].data["year"] = self.dat[0].data["year"]
        self.dat[-1].data["height"] = self.dat[0].data["height"]
        self.dat[-1].data["line_declination"] = \
            self.dat[0].data["line_declination"]
        if self.dat[-1].data["grad_data"]:
            self.dat[-1].data["height2"] = self.dat[0].data["height2"]
            self.dat[-1].data["d_sensor"] = self.dat[0].data["d_sensor"]
            self.dat[-1].inter_flag = False
        if self.dat[0].inter_flag:
            self.join_gridded()
        self.dat[-1].data_type = self.dat[0].data_type
        self.dat[-1].unit = self.dat[0].unit
        self.dat[-1].treatments = deepcopy(self.treatments)
        self.dat[-1].earth = self.dat[0].earth
        self.dat[-1].title = self.title_text
        self.dat[-1].xmin = self.dat[-1].x.min()
        self.dat[-1].xmax = self.dat[-1].x.max()
        self.dat[-1].ymin = self.dat[-1].y.min()
        self.dat[-1].ymax = self.dat[-1].y.max()
        self.dat_ori.append(deepcopy(self.dat[-1]))
        self.data_types.append(self.data_types[0])
        self.file_types.append(self.file_types[0])
        self.data_files.append(self.dat[-1].data["block_name"])
        self.data = deepcopy(self.dat[-1])
        self.data.prepare_gdata(original_fill=True)
        self.w.join.setEnabled(False)
        self.w.adjust.setEnabled(True)
        self.actual_plotted_file = len(self.dat) - 1
        self.plotActual()

    def join_gridded(self):
        self.dat[self.actual_plotted_file] = deepcopy(self.data)
        xmin = self.dat[0].x_inter.min()
        xmax = self.dat[0].x_inter.max()
        ymin = self.dat[0].y_inter.min()
        ymax = self.dat[0].y_inter.max()
        self.dx = self.dat[0].x_inter[1] - self.dat[0].x_inter[0]
        self.dy = self.dat[0].y_inter[1] - self.dat[0].y_inter[0]
        for i, d in enumerate(self.dat[1:-1]):
            if "+" in d.data["block_name"]:
                continue
            if not d.inter_flag:
                _ = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    f"Data set {d.data['block']} has not been gridded"
                    + "Interpolate the joint data set but be aware that\nif "
                    + "there are areas without data,\nthey may be interpolated"
                    + "with meaningless data.",
                    QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
                return
            xmin = min(xmin, d.x_inter.min())
            xmax = max(xmax, d.x_inter.max())
            ymin = min(ymin, d.y_inter.min())
            ymax = max(ymax, d.y_inter.max())
        self.x_inter = np.arange(xmin, xmax + self.dx / 2.0, self.dx)
        self.y_inter = np.arange(ymin, ymax + self.dy / 2.0, self.dy)
        nx = len(self.x_inter)
        ny = len(self.y_inter)
        self.sensor1_inter = np.zeros((ny, nx))
        self.sensor1_inter[:, :] = np.nan
        if self.dat[-1].data["grad_data"]:
            self.sensor2_inter = np.zeros((ny, nx))
            self.sensor2_inter[:, :] = np.nan
        for i, d in enumerate(self.dat[:-1]):
            dx = d.x_inter[1] - d.x_inter[0]
            dy = d.y_inter[1] - d.y_inter[0]
            if not (np.isclose(dx, self.dx) and np.isclose(dy, self.dy)):
                _ = QtWidgets.QMessageBox.warning(
                    None, "Warning",
                    f"Data set {d.data['block']}:\n\nGrid step is not like "
                    + f"others:\ndx = {dx}; required: {self.dx}\n"
                    + f"dy = {dy}; required: {self.dy}\n\n"
                    + "Interpolate all blocks with the same grid step and try"
                    + "again.\n\nJoin data sets is aborted",
                    QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
                self.n_blocks -= 1
                del self.dat[-1]
                return
            nx1 = int((d.x_inter.min() - xmin) / self.dx)
            nx2 = nx1 + len(d.x_inter)
            ny1 = int((d.y_inter.min() - ymin) / self.dy)
            ny2 = ny1 + len(d.y_inter)
            self.sensor1_inter[ny1:ny2, nx1:nx2] = np.copy(d.sensor1_inter)
            if self.dat[-1].data["grad_data"]:
                self.sensor2_inter[ny1:ny2, nx1:nx2] = np.copy(d.sensor2_inter)
        self.dat[-1].x_inter = np.copy(self.x_inter)
        self.dat[-1].y_inter = np.copy(self.y_inter)
        self.dat[-1].sensor1_inter = np.copy(self.sensor1_inter)
        if self.dat[-1].data["grad_data"]:
            self.grad_inter = (
                self.sensor2_inter - self.sensor1_inter) / \
                self.dat[-1].data["d_sensor"]
            self.dat[-1].sensor2_inter = np.copy(self.sensor2_inter)
            self.dat[-1].grad_inter = np.copy(self.grad_inter)
        self.dat[-1].inter_flag = True
        self.inter_flag = True
        self.data = deepcopy(self.dat[-1])
        self.data.prepare_gdata(original_fill=True)

    def getGeography(self):
        """
        Ask for file containing geography data, read data and plot them onto
        data map.

        Geography file has the following form:

        #Keyword (may be "#LINE", "#POINT" or "#END")
        If "#POINT", one line follows with x y text \
        If "#LINE", one line follows for every point defining the line x y

        The file must be finished with a line containing #END in the first
        four columns.
        Usually, points are towns and text the name of the town
        x and y must be given in the same coordinate system as the data.

        Returns
        -------
        None.

        """
        files = list(
            QtWidgets.QFileDialog.getOpenFileNames(
                None, "Select geography data file", "",
                filter="txt (*.txt) ;; all (*.*)"))
        if len(files) == 0:
            print("No file chosen, try again")
        if len(files[0]) < 1:
            print("getGeograpgy: No files read")
            return
        self.geography = io.read_geography_file(files[0][0])
        for d in self.dat:
            d.set_values(geography=self.geography)
            d.set_values(geography_flag=True)
        self.data.set_values(geography=self.geography)
        self.data.set_values(geography_flag=True)
        self.w.set_geography_flag(True)
        self.geography_flag = True
        self.w.plotGeo.setEnabled(True)
        self.plotActual()

    def oneOfTwo(self):
        """
        Extract every second line starting with line 1 (odd lines) or line 2
        (even lines). May be useful is there is a strong directional effect in
        magnetic data

        Returns
        -------
        None.

        """
        if self.treatments["odd lines"]:
            choice = ["even", "all"]
        elif self.treatments["even lines"]:
            choice = ["odd", "all"]
        else:
            choice = ["odd", "even"]
        results, okButton = dialog(
            ["Choose data of", choice], ["l", "r"], [None, 0], "Extract lines")
        if okButton:
            c = int(results[1])
            if self.treatments["odd lines"]:
                self.treatments["odd lines"] = False
                u.extract(self.data, choice[c])
                if c == 0:
                    self.treatments["even lines"] = True
                else:
                    self.treatments["even lines"] = False
            elif self.treatments["even lines"]:
                self.treatments["even lines"] = False
                if c == 0:
                    self.treatments["odd lines"] = True
                else:
                    self.treatments["odd lines"] = False
            else:
                if c == 0:
                    self.treatments["odd lines"] = True
                else:
                    self.treatments["even lines"] = True
            u.extract(self.data, choice[c])
            self.data.treatments = deepcopy(self.treatments)
        else:
            return
        self.plotActual()

    def readBaseData(self):
        """
        Read base station data. These data must be Geometrics PPM files

        Returns
        -------
        None.

        """
        data_files, _, _, _ = io.get_files(ftype="base")
        if len(data_files) > 0:
            for f in data_files:
                self.base_files.append(f)
                self.base = DataContainer(0)
                self.base.read_base(f, self.data.data["year"])
            self.w.basePlot.setEnabled(True)
            self.plotBase()

    def writeStn(self):
        """
        Writes actual (last treated interpolated data) to Geometrics stn file.
        The file name is "data-type"followed by date and hour of file
        production

        Returns
        -------
        None.

        """
        fname = u.file_name(self.data.data["type"], ".stn")
        self.data.write_geometrics(fname)

    def writeDat(self):
        """
        Writes actual (last treated not necessarily interpolated data) to
        Geometrics Surfer (.dat) format file.
        The file name is "data-type"followed by date and hour of file
        production

        Returns
        -------
        None.

        """
        fname = u.file_name(self.data.data["type"], ".dat")
        try:
            isinstance(self.sensor1_inter, np.ndarray)
            if self.data.data["grad_data"]:
                self.data.write_dat(fname)
            else:
                self.daa.write_dat(fname)
        except AttributeError:
            self.data.write_dat(fname)

    def writeGXF(self):
        file = u.file_name(self.data.data["type"], ".gxf")
        x0 = self.x_inter.min()
        y0 = self.y_inter.min()
        dx = self.x_inter[1] - self.x_inter[0]
        dy = self.y_inter[1] - self.y_inter[0]
        io.store_gxf(file, self.sensor1_inter, x0, y0, dx, dy)

    def saveBaseData(self):
        """
        Writes base station data to Geometrics G-856 .stn format file.
        The file name is "base"followed by date and hour of file production

        Returns
        -------
        None.

        """
        fname = u.file_name("base", ".stn")
        self.base.write_base(fname)

    def original(self):
        """
        If data were interpolated, go back to non-interpolated data, all
        earlier treatments are retained. If not, all treatments on data are
        undone.
        If no base station data had been read, but they were produced through
        spline fit of median values, those data are deleted.

        Returns
        -------
        None.

        """
        if not self.data.inter_flag:
            self.data = deepcopy(self.dat_ori[self.actual_plotted_file])
            self.treatments["odd lines"] = False
            self.treatments["even lines"] = False
            if len(self.base_files) == 0:
                try:
                    del self.w.base
                    del self.w.time_base
                except AttributeError:
                    pass
            self.treatments["diurnal"] = False
            self.treatments["clip"] = False
            self.treatments["justify_median"] = False
            self.treatments["justify_Gauss"] = False
            self.treatments["odd lines"] = False
            self.treatments["even lines"] = False
        self.w.originalData.setText("Restart with original data")
        self.w.fill.setEnabled(False)
        self.w.poleReduction.setEnabled(False)
        self.w.tiltAngle.setEnabled(False)
        self.w.lineFFT.setEnabled(False)
        self.w.Spector_2D.setEnabled(False)
        self.w.prolongation.setEnabled(False)
        self.w.analytic.setEnabled(False)
        self.w.gaussJustify.setEnabled(False)
        self.w.saveGXF.setEnabled(False)
        self.treatments["interpol"] = False
        self.treatments["nan_fill"] = False
        self.treatments["pole"] = False
        self.treatments["up"] = False
        self.data.treatments = deepcopy(self.treatments)
        self.inclination = self.inclination_ori
        self.declination = self.declination_ori
        self.earth.inc = self.inclination
        self.earth.dec = self.declination
        self.earth.earth_components()
        self.height = self.h_sensor + self.d_sensor
        self.data.set_values(
            earth=self.earth, treatments=self.treatments, inter_flag=False)
        self.plotActual()

    def plotOriginal(self):
        """
        Plot original data set again

        Returns
        -------
        None.

        """
        data = self.dat_ori[self.actual_plotted_file].data
        self.fig, self.ax = self.w.plot_triang(
            data, title=f"{self.data.data['title']}, ", percent=self.percent,
            mincol1=self.mincol1, maxcol1=self.maxcol1, mincol2=self.mincol2,
            maxcol2=self.maxcol2, mincolg=self.mincolg, maxcolg=self.maxcolg,
            grad_flag=self.gradient_flag, c=self.color)

    def plotActual(self):
        """
        Plot actual data set modified and interpolated or not

        Returns
        -------
        None.

        """
        title = f"{self.data.data['title']};\nTreatments:"
        ntreat = 0
        if self.data.inter_flag:
            for t in self.data.treatments.items():
                if t[1]:
                    if ntreat == 2:
                        title += "\n"
                    title += " " + t[0] + ","
                    ntreat += 1
            if title[-1] == ":":
                title += " None"
            if title[-1] == ",":
                title = title[:-1]

            self.fig, self.ax = self.w.plot_image(
                self.data, title=title, percent=self.percent,
                mincol1=self.mincol1, maxcol1=self.maxcol1,
                mincol2=self.mincol2, maxcol2=self.maxcol2,
                mincolg=self.mincolg, maxcolg=self.maxcolg,
                grad_flag=self.gradient_flag, c=self.color,
                dec=self.data.line_declination)
            self.w.setHelp(" ")
        else:
            for t in self.data.treatments.items():
                if t[0] in ("interpol", "nan_fill", "pole", "up"):
                    continue
                if t[1]:
                    title += " " + t[0] + ","
            if title[-1] == ":":
                title += " None"
            if title[-1] == ",":
                title = title[:-1]
            self.fig, self.ax = self.w.plot_triang(
                self.data, title=title, percent=self.percent,
                mincol1=self.mincol1, maxcol1=self.maxcol1,
                mincol2=self.mincol2, maxcol2=self.maxcol2,
                mincolg=self.mincolg, maxcolg=self.maxcolg,
                grad_flag=self.gradient_flag, c=self.color)
            self.w.setHelp(" ")

    def plot_gradient(self):
        """
        Activate or desactivate plot of vertical gradient maps

        Returns
        -------
        None.

        """
        self.gradient_flag = not self.gradient_flag
        if self.gradient_flag:
            self.w.plotGradient.setChecked(True)
        else:
            self.w.plotGradient.setChecked(False)
        self.plotActual()

    def plot_geography(self):
        """
        Activate or deactivate plot of geographical information

        Returns
        -------
        None.

        """
        self.w.toggle_geography_flag()
        for d in self.dat:
            d.set_values(geography_flag=self.w.geography_flag)
        self.data.set_values(geography_flag=self.w.geography_flag)
        self.plotActual()

    def plot_lineaments(self):
        """
        Activate or deactivate plot of measured lineaments

        Returns
        -------
        None.

        """
        self.w.plotLin_flag = not self.w.plotLin_flag
        for d in self.dat:
            d.set_values(plotLin_flag=self.w.plotLin_flag)
        self.data.set_values(plotLin_flag=self.w.plotLin_flag)
        self.plotActual()

    def plot_grid(self):
        """
        Activate or deactivate plot of grid

        Returns
        -------
        None.

        """
        self.w.grid_flag = not self.w.grid_flag
        self.grid_flag = not self.grid_flag
        for d in self.dat:
            d.set_values(grid_flag=self.w.grid_flag)
        self.plotActual()

    def plot_points(self):
        """
        Activate or deactivate plotting of measurement point positions

        Returns
        -------
        None.

        """
        self.w.point_flag = not self.w.point_flag
        self.point_flag = not self.point_flag
        self.plotActual()

    def plot_line(self):
        """
        Plot data of a specfic (mouse-chose) line

        Returns
        -------
        None.

        """
        self.plotLine(plot_flag=True)

    def plotLine(self, plot_flag=True):
        """
        Plot one line out of the data set.
        The user is first asked to choose one line by mouse click. This line
        is shown in a floating window.
        Erroneous data may be erased by clicking the mouse wheel at the
        beginning and at the end of the zone to be erased. The program
        recognizes automatically whether first or second sensor has been
        clicked. Left mouse click changes line to the left/below, right mouse
        click goes to the line to the right/above.
        A click outside the two coordinate systems stops the module and deletes
        the flowting window.

        Parameters
        ----------
        plot_flag : bool, optional. Default: True
            If true, data along line are plotted and returned, if False, data
            are returned but nor plotted.

        Returns
        -------
        pos : 1D numpy float array
            contains the coordinates of tha data pointsalong the chosen line
        pos_line : float
            position of the chosen line
        topo : 1D numpy float array
            Topography along the line if given. Positive downwards, i.e., in
            general topography is zero, if not given, or negative
        z1 : 1D numpy float array
            Height of receiver 1 along the line. Constant value equal to the
            height given in file *.config if no topography given. Else, it is
            topography plus the height given or the height of the receiver
            above sea level if given in the data file. Positive downwards.
        z2 : 1D numpy float array
            Height of receiver 2 along the line (z1 plus the height difference
            given in file *.config). If only one sensor exists, z2 = z1.
        s1 : 1D numpy float array
            Measured data along the line of sensor 1
        s2 : 1D numpy float array
            Measured data along the line of sensor 2. If only one sensor, s2 is
            np.array([None])
        direction : str
            May be "N" or "E", depending on whether the left or right mouse
            button was pressed

        """
        direction = ""
        if self.inter_flag:
            self.w.setHelp(
                "Click left mouse button to choose line in Y "
                + "direction or right button for X-direction")
        else:
            if self.direction == 0:
                self.w.setHelp("Click mouse button to choose a line in Y"
                               + "direction")
            else:
                self.w.setHelp("Click mouse button to choose a line in X"
                               + "direction")
# Wait for mouse click to choose line to be plotted first
        while True:
            event = self.w.get_mouse_click(self.fig)
            if event.name == "button_press_event":
                if event.inaxes:
                    break
        pos, pos_line, topo, z1, z2, s1, s2, direction = \
            self.data.plot_line(plot_flag, event, title=self.title_text)
        self.plotActual()
        self.w.setHelp(" ")
        return pos, pos_line, topo, z1, z2, s1, s2, direction

    def changeColorScale(self):
        """
        Change the limiting quantiles for creating of color scales.

        The values below self.percent and above (1-self.percent) are clipped

        Returns
        -------
        None.

        """
        cols = ["rainbow", "seismic", "viridis", "special mag"]
        if self.w.grad_data:
            results, okButton = dialog(
                ["If min==max, quantile is used\n"
                 + "If quantile is also 0, min and max values of arrays\n"
                 + "If min!=max, these values set the color scale limits",
                 "Cliping quantile for plotting",
                 f"Minimum of color scale sensor1 [{self.unit}]",
                 f"Maximum of color scale sensor1 [{self.unit}]",
                 "_______________________________",
                 f"Minimum of color scale sensor2 [{self.unit}]",
                 f"Maximum of color scale sensor2 [{self.unit}]",
                 "_______________________________",
                 f"Minimum of color scale gradient [{self.unit}/m]",
                 f"Maximum of color scale gradient [{self.unit}/m]",
                 "Color map", cols],
                ["l", "e", "e", "e", "l", "e", "e", "l", "e", "e", "l", "b"],
                [None, self.percent, self.mincol1, self.maxcol1, None,
                 self.mincol2, self.maxcol2, None, self.mincolg, self.maxcolg,
                 0], "Color scale limits")
        else:
            results, okButton = dialog(
                ["If min==max, quantile is used\nIf quantile is also 0, min "
                 + "and max values of arrays\nIf min!=max, these values set "
                 + "the color scale limits" "Cliping quantile for plotting",
                 f"Minimum of color scale [{self.unit}]",
                 f"Maximum of color scale [{self.unit}]", "Color map", cols],
                ["l", "e", "e", "e", "l", "b"],
                [None, self.percent, self.mincol1, self.maxcol1, 0],
                "Color scale limits")
        if okButton:
            self.percent = float(results[1])
            self.mincol1 = float(results[2])
            self.maxcol1 = float(results[3])
            if self.w.grad_data:
                self.mincol2 = float(results[5])
                self.maxcol2 = float(results[6])
                self.mincolg = float(results[8])
                self.maxcolg = float(results[9])
                self.color = cols[int(results[11])]
            else:
                self.color = cols[int(results[5])]
            self.plotActual()
        else:
            print(f"\nClipping quantile left at {self.percent:0.3f}\n"
                  + f"    minimum color: {self.mincol1:0.1f} {self.unit}"
                  + f"    maximum color: {self.maxcol1:0.1f} {self.unit}")

    def plotBase(self):
        """
        Plot base station data as function of time (seconds in year).
        The user may erase erroneous data by clicking the mouse wheel at the
        beginning and at the end of the zone to be erased.
        A click with any other mouse button finishes the module and closes the
        floating window.

        Returns
        -------
        None.

        """
        try:
            self.fig_base.clf()
        except (NameError, AttributeError):
            pass
        base = self.base.base
        base = FP.base_plot(base)

    def plot_median(self):
        """
        Wrapper to plot medians of all lines

        """
        self.data.plot_median()

    # def zooming(self):
    #     pass

    # def zoomOut(self):
    #     pass

    # def zoomIn(self):
    #     pass

    # def zoomInitial(self):
    #     pass

    def save_plot(self):
        """
        Function saves plot inside actual window into a png file
        The file name is prefix_date_time.png
        Prefix depends on the actual image type.

        Returns
        -------
        None.

        """
        now = datetime.now()
        c_time = now.strftime("%H-%M-%S")
        today = date.today()
        d1 = today.strftime("%Y-%m-%d")
        fname = f"{self.data.data['type']}_" + f"{d1}_{c_time}.png"
        self.fig.savefig(fname)

    def diurnal(self):
        """
        If no base station data have been read create diurnal variation
        by adjusting a spline of degree deg to the median values as function
        of time.

        """
        if self.treatments["interpol"]:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "Since during interpolation timing information is lost,\n "
                + "diurnal variations can only be corrected on ungridded data."
                + "\n\nUse Utilities->Restart with ungridded data and try "
                + "again.",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return

        if len(self.base_files) > 0:
            deg = 3
            self.base_flag = True
        else:
            results, okButton = dialog(
                ["No base station data exist.\nFit polygone to medians or "
                 + "cancel", "Degree of polynom"], ["l", "e"], [None, 5],
                "Diurnal variation by curve fit")
            if not okButton:
                print("\nDiurnal correction not applied")
            deg = int(results[1])
            self.base = DataContainer(0)
            self.base.base_ini()
            self.base_flag = False
        result = u.diurnal_correction(
            self.data, self.base.base, base_flag=self.base_flag, degree=deg)
        if result:
            self.treatments["diurnal"] = True
            self.data.set_values(treatments=self.treatments)
        self.w.basePlot.setEnabled(True)
        self.plotActual()

    def clean(self):
        """
        Wrapper to delete erroneous data

        Returns
        -------
        None.

        """
        self.data.clean()
        self.treatments["clip"] = True
        self.plotActual()

    def justify_median(self):
        """
        Wrapper to reduce directional effect by median adjustment

        """
        self.data.justify_median()
        self.treatments["justify_median"] = True
        self.plotActual()

    def justify_gauss(self):
        """
        Wrapper to reduce directional effect by Gaussian adjustment
        see Masoudi et al., J. Geophys. Eng., 2023

        Returns
        -------
        None.

        """
        self.data.justify_gauss()
        self.treatments["justify_Gauss"] = True
        self.plotActual()

    def interpol(self):
        """
        Interpolate data within the measurement lines onto a regular grid.
        No data are extrapolated, i.e. if a line starts later or finishes
        earlier than a regular grid, missing grid points are set to nan

        Returns
        -------
        None.

        """
        self.data.interpol()
        self.dat[self.actual_plotted_file] = deepcopy(self.data)
        self.treatments["interpol"] = True
        if "m" in self.data.data_type:
            self.w.poleReduction.setEnabled(True)
        self.w.tiltAngle.setEnabled(True)
        self.w.lineFFT.setEnabled(True)
        self.w.Spector_2D.setEnabled(True)
        self.w.prolongation.setEnabled(True)
        self.w.analytic.setEnabled(True)
        self.w.gaussJustify.setEnabled(True)
        self.w.originalData.setText("Restart with ungridded data")
        self.plotActual()

    def nan_fill(self):
        """
        Fill nan values by interpolation of data in the direction perpendicular
        to the measurement direction (it is supposed that if a line is not
        complete, nearby ones will be). Extrapolation will be done (one wants
        to create a complete grid) and different possibilities exist (mainly
        spline or constant). Spline is often very risky.

        Returns
        -------
        None.

        """
        self.data.nan_fill()
        self.plotActual()

    def reduce_pole(self):
        """
        Wrapper for pole reduction is done.

        Returns
        -------
        None.

        """
        self.data.reduce_pole()
        self.plotActual()

    def spector(self):
        """
        Wrapper for calculation of 2D Spector&Grant depths for all lines

        """
        self.data.spector()

    def spector2D(self):
        """
        Calculate depth of random sources with formula of (Spector and Grant,
        Geophysics, 1970) in 2D.

        Choose first a window length. Spectral coefficients are averaged over
        radial equidistant coefficients. Depths are calculated by fitting two
        lines to logarithmic spectrum. The break point between the two lines is
        searched between the 4th and the 10th spectral coefficient.
        Results of all lines are saved in file spector.dat.

        The window length should be defined such that the number of Fourier
        coefficients is at least 8:
        (n = window_length/(2*max(dx,dy)), dx, dy, step sizes defined during
        interpolation)

        Results are stored in file spector2D_<data_type>.dat; Data type may be
        "magnetic" or "gravity".

        """
        self.data.spector2D()

    def tilt(self):
        """
        Calculate tilt angle (Miller & Singh, JAG, 1994)

        Returns
        -------
        None.

        """
        self.lineaments = self.data.tilt()
        self.nlineaments = len(self.lineaments)
        if self.nlineaments:
            self.w.plotLineaments.setEnabled(True)
            self.w.plotLin_flag = True
            # self.plot_lineaments()
        else:
            self.w.plotLineaments.setEnabled(False)
            self.w.plotLin_flag = False
        self.data.set_values(plotLin_flag=self.w.plotLin_flag)
        for d in self.dat:
            d.set_values(plotLin_flag=self.w.plotLin_flag)
        self.data.set_values(plotLin_flag=self.w.plotLin_flag)
        self.save_lineaments()
        self.plotActual()

    def continuation(self):
        """
        Wrapper to calculate field at higher or lower altitude
        """
        self.dz = self.data.continuation()
        self.treatments["up"] = True
        self.height += self.dz
        self.plotActual()

    def analytic_signal(self):
        """
        Calculate analytic signal (Nabighian, Geophysics, 1972)

        Returns
        -------
        None.

        """
        self.data.analytic_signal()

    def inver2D(self):
        """
        Do a 2D inversion along an interactively chosen line

        Returns
        -------
        None.

        """
        earth = deepcopy(self.earth)
        earth.dec -= self.data.data["line_declination"]
        data = []
        x = []
        y = []
        z = []
        data_type = self.data_types[self.actual_plotted_file][0]
        xx, yy, topo_line, z_line1, z_line2, s1, s2, direction = \
            self.plotLine(plot_flag=False)
        if self.data.topo_flag:
            zz = z_line1
        else:
            topo_line = np.zeros_like(xx)
            zz = np.ones_like(xx) * self.data.data["height"]
        if direction in ("N", "S", 0.0, 180.0):
            file = f"Inver2D_line_{int(yy)}m-E"
        else:
            file = f"Inver2D_line_{int(yy)}m-N"
            earth.dec -= 90.0
        data.append(s1)
        x.append(xx)
        y.append(None)
        if self.data.topo_flag:
            z.append(zz)
        else:
            z.append(-zz)
        if s2[0] is not None:
            x.append(xx)
            y.append(None)
            if self.data.topo_flag:
                z.append(z_line2)
            else:
                z.append(-z_line2)
            data.append(s2)
        inv = inversion(self.data, data, x, y, z, topo=topo_line, earth=earth,
                        data_type=data_type, line_pos=yy, direction=direction,
                        dim=2)
# Define inversion parameters
        ret = inv.get_inversion_parameters(data_type)
        if not ret:
            return
# Set overall data and parameter variances
        ret = inv.get_variances()
        if not ret:
            return
# Get area of initial prisms and extract data to be inverted
        ret = inv.get_area2D()
        if not ret:
            return
        while True:
            inv.run_inversion()
            ret = inv.show_results2D(file)
            if not ret:
                break
            ret = inv.get_inversion_parameters(data_type)
            if not ret:
                break
            ret = inv.get_variances()
            if not ret:
                break
        inv.save_model()

    def inver3D(self):
        """
        Do a 3D inversion of data shown on screen

        Data must be gridded.

        It is possible to reduce the area to be inverted and the number of
        data points, taking one point out of n.

        The user will be asked interactively for the area to be inverted, for
        general inversion control parameters like the maximum number of
        iterations, other stopping conditions and smoothing and data and
        parameter variances.

        Returns
        -------
        None.

        """
        earth = deepcopy(self.earth)
        earth.dec -= self.data.data["line_declination"]
        if self.data.inter_flag:
            data_type = self.data_types[self.actual_plotted_file][0]
            data = []
            x = []
            y = []
            z = []
            data.append(self.data.sensor1_inter)
            xx, yy = np.meshgrid(self.data.x_inter, self.data.y_inter)
            x.append(xx)
            y.append(yy)
            z.append(self.data.z_fill)
            if self.data.data["grad_data"]:
                data.append(self.data.sensor2_inter)
                x.append(xx)
                y.append(yy)
                z.append(self.data.z_fill + self.data.data["d_sensor"])
            inv = inversion(self.data, data, x, y, z, earth=earth,
                            data_type=data_type, dim=3)
# Define inversion parameters
            ret = inv.get_inversion_parameters(data_type)
            if not ret:
                return
# Set overall data and parameter variances
            ret = inv.get_variances()
            if not ret:
                return
# Get area of initial prisms and extract data to be inverted
            ret = inv.get_area3D()
            if not ret:
                return
# Do inversion
            while True:
                inv.run_inversion()
                ret = inv.show_results3D()
                if not ret:
                    break
                ret = inv.get_inversion_parameters(data_type)
                if not ret:
                    break
                ret = inv.get_variances()
                if not ret:
                    break
            inv.save_model()

        else:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "For 3D inversion, data must be interpolated onto a "
                + "regular grid.\n\nUse Utilities->Interpolate\n"
                + "and start inversion again",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return

    def synthetic_model(self):
        """
        Define a 3D model composed of rectangular vertical prisms and calculate
        its effect. By default, the same type of data is calculated as the one
        of the read data (magnetic vs gravity and if magnetic, the same
        parameters for the Earth's field)')

        Returns
        -------
        None.

        """
        data_type = self.data_types[self.actual_plotted_file][0]
        x, y, z, sus, rem, rem_i, rem_d, rho = io.read_synthetic_model()
        if x is None:
            nprism = 0
            x1 = []
            x2 = []
            y1 = []
            y2 = []
            z1 = []
            z2 = []
            sus = []
            rem = []
            rem_i = []
            rem_d = []
            rho = []
            if "m" in data_type:
                s = 0.001
                r = 0.0
            else:
                s = 0.0
                r = 100.0
            while True:
                results, okButton = dialog(
                    [f"Prism {nprism+1}\n\n", "xmin [m]", "xmax [m]",
                     "ymin [m]", "ymax [m]", "zmin [m]", "zmax [m]",
                        "_________________________________",
                        "susceptibility [SI]", "remanence intensity [A/m]",
                        "remanence declination [°]",
                        "remanence inclination [°]", "density [kg/m3]"],
                    ["l", "e", "e", "e", "e", "e", "e", "l", "e", "e", "e",
                     "e", "e"],
                    ["b", 0.0, 50.0, 0.0, 50.0, 1.0, 5.0, None, s, 0.0, 0.0,
                     0.0, r], "Synthetic model")
                if not okButton:
                    if nprism == 0:
                        print("No prism defined, synthetic model calculation "
                              + "aborted")
                        return
                    break
                nprism += 1
                x1.append(float(results[1]))
                x2.append(float(results[2]))
                y1.append(float(results[3]))
                y2.append(float(results[4]))
                z1.append(float(results[5]))
                z2.append(float(results[6]))
                sus.append(float(results[8]))
                rem.append(float(results[9]))
                rem_i.append(float(results[10]))
                rem_d.append(float(results[11]))
                rho.append(float(results[12]))
            x = np.zeros((len(x1), 2))
            x[:, 0] = np.array(x1)
            x[:, 1] = np.array(x2)
            y = np.zeros((len(x1), 2))
            y[:, 0] = np.array(y1)
            y[:, 1] = np.array(y2)
            z = np.zeros((len(x1), 2))
            z[:, 0] = np.array(z1)
            z[:, 1] = np.array(z2)
            sus = np.array(sus)
            rem = np.array(rem)
            rem_i = np.array(rem_i)
            rem_d = np.array(rem_d)
            rho = np.array(rho)
            with open("synthetic_model.txt", "w") as fo:
                for i, s in enumerate(sus):
                    fo.write(f"{x[i, 0]} {x[i, 1]} {y[i, 0]} {y[i, 1]} "
                             + f"{z[i, 0]} {z[i, 1]} {s} {rem[i]} {rem_i[i]} "
                             + f"{rem_d[i]} {rho[i]}\n")
        nprism = len(sus)
        xmin = self.data.xmin
        xmax = self.data.xmax
        ymin = self.data.ymin
        ymax = self.data.ymax
        dx = max(np.round((xmax - xmin) / 50.0, 0), 0.5)
        dy = max(np.round((ymax - ymin) / 50.0, 0), 0.5)
        labels = ["xmin [m]", "xmax [m]", "dx [m]", "__________________",
                  "ymin [m]", "ymax [m]", "dy [m]"]
        types = ["e", "e", "e", "l", "e", "e", "e"]
        values = [xmin, xmax, dx, None, ymin, ymax, dy]
        if "m" in data_type:
            labels += ["__________________\nif height2==height1 or"
                       + "height2==None:\n   only 1 sensor",
                       "height sensor 1", "height sensor 2"]
            types += ["l", "e", "e"]
            values += [None, -self.dat[-1].data["height"],
                       -self.dat[-1].data["height2"]]
        results, okButton = dialog(labels, types, values, "Calculation points")
        if not okButton:
            print("Calculation of synthetic model aborted")
            return
        xmin = float(results[0])
        xmax = float(results[1])
        dx = float(results[2])
        ymin = float(results[4])
        ymax = float(results[5])
        dy = float(results[6])
        if len(labels) > 7:
            height1 = -float(results[8])
            if results[9].lower() == "none":
                height2 = height1
            else:
                height2 = -float(results[9])
        else:
            height1 = 0.0
            height2 = 0.0
        if np.isclose(xmin, xmax) or np.isclose(ymin, ymax):
            dim = 2
        else:
            dim = 3
        earth = deepcopy(self.earth)
        earth.dec -= self.data.data["line_declination"]
        xx = np.arange(xmin, xmax + dx / 2.0, dx)
        yy = np.arange(ymin, ymax + dy / 2.0, dy)
        xd, yd = np.meshgrid(xx, yy)
        d_shape = xd.shape
        ndat1 = d_shape[0] * d_shape[1]
        xdat = []
        ydat = []
        zdat = []
        data = []
        xdat.append(xd)
        ydat.append(yd)
        zdat.append(np.ones_like(xd) * height1)
        data.append(np.zeros_like(xd))
        if height2 != height1:
            data.append(np.zeros_like(xd))
            xdat.append(xd)
            ydat.append(yd)
            zdat.append(np.ones_like(xd) * height2)
        self.sPrism = PP(earth)
        for i in range(nprism):
            if "m" in data_type.lower():
                typ = "M"
            else:
                typ = "G"
            self.sPrism.add_prism(
                x[i], y[i], z[i], sus[i], rem[i], rem_i[i], rem_d[i], rho[i],
                0, typ=typ)
        topo = np.zeros_like(xdat[0])
        inv = inversion(self.data, data, xdat, ydat, zdat, earth=earth,
                        topo=topo, data_type=data_type, dim=dim, act="D")
        inv.mPrism = self.sPrism
        inv.params = np.zeros(nprism + 1)
        if "m" in data_type:
            if abs(sus).max() > 0.0:
                inv.sus_inv = True
                inv.params[:nprism] = sus
            if abs(rem).max() > 0.0:
                inv.rem_inv = True
                inv.params[:nprism] = rem
        else:
            inv.rho_inv = True
            inv.params[:nprism] = rho
            inv.params
        inv.max_iter = 0
        if height1 != height2:
            inv.x = np.concatenate((xdat[0].flatten(), xdat[1].flatten()))
            inv.y = np.concatenate((ydat[0].flatten(), ydat[1].flatten()))
            inv.z = np.concatenate((zdat[0].flatten(), zdat[1].flatten()))
            inv.data = np.zeros_like(inv.x)
        inv.run_inversion()
        dat = DataContainer(0)
        if height1 == height2:
            dat.store_gxf(os.path.join(inv.folder, "synthetic_data.gxf"),
                          inv.data_mod.reshape(d_shape), xmin, ymin, dx, dy)
        else:
            dat.inter_flag = True
            dat.sensor1_inter = inv.data_mod[:ndat1].reshape(d_shape)
            dat.sensor2_inter = inv.data_mod[ndat1:].reshape(d_shape)
            dat.z_inter = zdat[0]
            dat.x_inter = xx
            dat.y_inter = yy
            dat.grad_data = True
            dat.write_dat(os.path.join(inv.folder, "synthetic_data.dat"), True)
            with open(os.path.join(inv.folder, "synthetic_data.config"),
                      "w") as fo:
                fo.write("MGWIN\n")
                fo.write(f"{data_type}\n")
                fo.write("synthetic model\n")
                fo.write(f'{self.dat[-1].data["line_declination"]}\n')
                fo.write(f"{-height1}\n")
                fo.write(f"{-height2}\n")
                fo.write("0\n")
                fo.write(f"{self.earth.f}\n")
                fo.write(f"{self.earth.inc}\n")
                dec = self.earth.dec
                fo.write(f'{dec+self.dat[-1].data["line_declination"]}\n')
        inv.show_synthetic()

    def Handler(self, signal_received, frame):
        """
        Handles CTRL-C key stroke

        Parameters
        ----------
        signal_received : CTRL-C keystroke
        frame : No idea

        Returns
        -------
        None.

        """
        self.close_app()

    def save_lineaments(self):
        """
        If lineaments were picked, save the information into file
        lineaments.dat

        The file has the format:
        #key (including the character #)
        x y

        Actually, key may be "magnetic tilt" or "gravity tilt", meaning
        lineaments having been traced on tilt angle maps of gravity or
        megnetic data.

        x y line is repeated for every point defining a lineament.
        File is finished with "#END"

        Returns
        -------
        None.

        """
        if self.nlineaments > 0:
            with open("lineaments.dat", "w", encoding="utf-8") as fo:
                for lin in self.lineaments.values():
                    fo.write(f"#{lin['type']}\n")
                    for i in range(len(lin["x"])):
                        fo.write(f"{lin['x'][i]:0.1f}  {lin['y'][i]:0.1f}\n")
                fo.write("#END\n")

    def close_app(self):
        """
        Finishes application:

        Stores picks into file pick.dat
        Stores information concerning possibly modified traces
        (muted or sign-inversed)
        Deletes unneeded folder if tomography was calculated
        Closes window
        """
        choice = QtWidgets.QMessageBox.question(
            None, "Confirm", "Are you sure?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("\nApplication finished.\n\n"
                  + "Close console if you are working with Spyder")
            self.w.close()
            QtWidgets.QApplication.quit()
            return True
        return False
