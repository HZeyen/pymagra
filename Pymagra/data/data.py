# -*- coding: utf-8 -*-
"""
Last modified Feb 24, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         University Paris-Saclay, France

Contains class DataContainerto treat and plot potential fiel data.
"""

import sys
from copy import deepcopy
import gc
from datetime import datetime
from PyQt5 import QtWidgets
import numpy as np
from ..in_out.dialog import dialog
from ..in_out import io
from ..in_out import communication as comm
from .geometrics import Geometrics
from ..utilities import utilities as u
from ..utilities import transforms as trans
from ..plotting import floating_plots as FP
from ..plotting.new_window import newWindow


class DataContainer:
    """
    Class contains methods for data management in program PyMaGra

    Contains the following methods:
       - __init__
       - set_values : set certain attributes defined outside the class
       - read_geometrics : Read data from Geometrics .stn format file
       - correct_time : Correct measurement times
       - write_geometrics : Write data into Geometrics .stn format file
       - write_dat : Write data into Geometrics .dat (Surfer) format file
       - read_txt : Read data from simple text file
       - read_gxf : Read gridded data from gxf format file
       - store_gxf : Store gridded data into gxf format file
       - read_BRGM_flight : Read data from a BRGM aeromagnetid data file
       - get_line : Get data from a line out of class Geometrics data
       - lines : Transform data from class Geometrics to class DataContainer
       - read_base : Read data from a Geometrics base station file
       - base_ini : InitializeGeometrics class instance for base station data
       - write_base : Write base station data into Geometrics .stn file
       - interpol : Interpolate data onto a regular grid (convex area)
       - nan_fill : Extrapolate data onto full rectangular grid
       - clean : Mute (set to nan) data outside user-defined limits
       - justify_median : Eliminate directional effect by adjusting medians
       - justify_gauss : Eliminate directional effect by adjusting Gaussion
         statistics
       - plot_median : Plot median values of all lines as function of sensor
         and measurement direction
       - reduce_pole : Reduce magnetic data to pole
       - spector : Estimate average source depths of each line using spectral
         decay
       - spector2D : Estimate source depth using spectral decay using sliding
         windows
       - tilt : Calculate and plot tilt angles
       - analytic_signal : Calculate plot and invert analytic signal data
       - continuation : Continue data to another measurement
       - delete_data : Mute data of a single line between user-given
         coordinates
       - get_linedata : Get all data along a line from different measurement
         blocks
       - plot_line : Extract, plot and modify data of a given line

    """

    def __init__(self, n_block):
        """
        Initialisation of class Data

        Parameters
        ----------
        n_block : int
            Number of data set read.
        window : object of class plotting.plot
            Main window to plot in

        Returns
        -------
        None.

        """
        self.sensor1 = []
        self.sensor2 = []
        self.sensor1_ori = []
        self.sensor2_ori = []
        self.gdata = []
        self.base = []
        self.grad = np.array([])
        self.grad_ori = np.array([])
        self.x_inter = np.array([])
        self.y_inter = np.array([])
        self.x = []
        self.y = []
        self.z = []
        self.topo = []
        self.dispo = 0
        self.sensor1_inter = np.zeros((1, 1))
        self.sensor2_inter = np.zeros((1, 1))
        self.time = []
        self.segments = {}
        self.geography = {}
        self.grad_data = False
        self.inter_flag = False
        self.nan_flag = False
        self.lineament_flag = False
        self.geography_flag = False
        self.plotLin_flag = False
        self.topo_flag = False
        self.grid_data = False
        self.d_sensor = 0.9
        self.h1_sensor = 1.4
        self.h2_sensor = 0.4
        self.line_declination = 0.0
        self.inclination = 60.
        self.declination = 0.
        self.d_inter = 0.2
        self.data = {}
        self.treatments = {}
        self.lineaments = {}
        self.n_block = n_block
        self.n_lines = 0
        self.n_data = 0
        self.direction = 0
        self.dx = 0.0
        self.dy = 0.0
        self.line_choice = "all"
        self.xmin = 1.E10
        self.xmax = -1.E10
        self.ymin = 1.E10
        self.ymax = -1.E10
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
        self.config_file = ""
        self.file_name = ""
        self.file_type = ""
        self.data_type = ""
        self.unit = ""
        self.gdata = Geometrics()

    def set_values(self, line_declination=None, h_sensor=None, h2_sensor=None,
                   d_sensor=None, dispo=None, block=None, block_name=None,
                   typ=None, inter_flag=None, treatments=None, fconfig=None,
                   file=None, ftype=None, dtype=None, earth=None,
                   lineaments=None, plotLin_flag=None, geography=None,
                   geography_flag=None, grid_flag=None):
        """
        Sets certain attributes of DataContainer from outside.
        Only values not set to None are redefined.

        Parameters
        ----------
        line_declination : float, optional; default: None
            Direction of local Y axis with respet to North
        h_sensor : float, optional; default: None
            Height of sensor1 above ground [m]
        h2_sensor : float, optional; default: None
            Height of sensor2 above ground [m]
        d_sensor : float, optional; default: None
            Distance between sensors (h_sensor - h2_sensor)
        dispo : int, optional; default: None
            If 0: Vertical disposition of sensors, if 1: horizontal disposition
        block : int, optional; default: None
            Number of block (file, data set)
        block_name : str, optional; default: None
            Name of block
        typ : str, optional; default: None
            Data type, may be "magnetic" or "gravity"
        inter_flag : bool, optional; default: None
            If True, data are interpolated onto a regular grid, else: original
            data.
        treatments : dict, optional; default: None
            Dictionary with booleans indicating the treatments having been
            effectuated. Contains the following items:

            - "diurnal" (correction of diurnal variations)
            - "clip" (Data were clipped)
            - "justify_median" (directional effects were reduced adjusting
              medians)
            - "justify_Gauss" (directional effects were reduced adjusting
              Gaussian statistics)
            - "interpol" (Date were interpolated onto a regular grid)
            - "nan_fill" (Nans in big holes or in corners were filled by
              extrapolation)
            - "pole" (Magnetic data were reduced to pole)
            - "odd lines" (only odd lines are chosen)
            - "even lines" (only even lines are chosen)
            - "up" (data were up-or downward continued)

        fconfig : str, optional; default: None
            File with configuration information
        file : str, optional; default: None
            Data file name
        ftype : str, optional; default: None
            File type. May be "GEOMETRICS", "MGWIN", "BRGM" or "GXF"
        dtype : str, optional; default: None
            Data type. May be "magnetic" or "gravity"
        earth : object of class earth.Earth_mag, optional; default: None
            Parameters of the Earth's magnetic field
        lineaments : dictionary
            Contains lineaments, usually obtained interactively from tilt maps
        plotLin_flag : bool
            If True, lineaments are plotted on maps, else plotting is avoided
        geography : dictionary
            Contains geography information
        geography_flag : bool
            If True, geography information is plotted on maps, else not
        grid_flag : bool
            If True, axes grid lines are plotted

        Returns
        -------
        None.

        """
        if line_declination is not None:
            self.line_declination = line_declination
        if h_sensor is not None:
            self.h1_sensor = h_sensor
        if h2_sensor is not None:
            self.h2_sensor = h2_sensor
        if d_sensor is not None:
            self.d_sensor = d_sensor
        if dispo is not None:
            self.dispo = dispo
        if block is not None:
            self.data["block"] = block
        if block_name is not None:
            self.data["block_name"] = block_name
        if typ is not None:
            self.data["type"] = typ
            if "m" in typ:
                self.unit = "nT"
            else:
                self.unit = "mGal"
        if inter_flag is not None:
            self.inter_flag = inter_flag
        if treatments is not None:
            self.treatments = deepcopy(treatments)
        if fconfig is not None:
            self.config_file = fconfig
        if file is not None:
            self.file_name = file
        if ftype is not None:
            self.file_type = ftype
        if dtype is not None:
            self.data_type = dtype
        if earth is not None:
            self.earth = earth
            self.inclination = earth.inc
            self.declination = earth.dec
        if lineaments is not None:
            self.lineaments = lineaments
        if plotLin_flag is not None:
            self.plotLin_flag = plotLin_flag
        if geography is not None:
            self.geography = geography
        if geography_flag is not None:
            self.geography_flag = geography_flag
        if grid_flag is not None:
            self.grid_flag = grid_flag

    def read_geometrics(self, file, height1=None, height2=None, disp=None,
                        dec=None, title=None):
        """
        Read Geometrics .stn or .dat file (G-858 instrument)

        Parameters
        ----------
        file : str
            Name of data file.
        height1 : float, optional. Default: None
            Height of sensor 1 above ground (meters)
        height2 : float, optional. Default: None
            Height of sensor 2 above ground (meters)
        disp : int, optional. Default: None
            Disposition of sensors if there are two sensors:
            0: vertical disposition, 1: horizontal disposition
        dec : float, optional. Default: None
            Direction of y axis with respect to North (magnetic north for
            magnetic data, geographic north for gravity).
        title : str, optional. Default None
            Title appearing on most plots

        Returns
        -------
        data :  Dictionary with keys equal to line numbers (starting at 0)
            Each line is itself a dictionary with the following entries:
            Key is line number

            - "s1": Numpy float array with data of sensor 1
            - "s2": Numpy float array with data of sensor 2
              If only data of one single sensor were measured, "s2" contains
              only one zero.
            - "x":  Numpy float array with E-W coordinates of data points
            - "y":  Numpy float array with N-S coordinates of data points
            - "grad_flag" bool. True if 2 sensors were used, False if only
              one sensor
            - "mask": bool, True if line should be plotted, False if excluded
              from plotting. Initially set to True

        The original data are stored in class geometrics.Geometrics. See file
        geometrics.py for documentation

        """
# If values for sensor height and sensor dispodition are not known, open
# dialogue box for interactive definition
        if height1 is None:
            ret, self.h1_sensor, self.h2_sensor, self.dispo, \
                self.line_declination, self.title = comm.get_geometry(
                    file, h1=self.h1_sensor, h2=self.h2_sensor,
                    dispo=self.dispo+1, dec=self.line_declination,
                    title="Magnetic data")
            if not ret:
                print(f"file {file} not read")
                return False
        else:
            self.h1_sensor = height1
            self.h2_sensor = height2
            self.d_sensor = height1 - height2
            self.dispo = disp
            self.line_declination = dec
            self.title = title
# Read geometrics data. Depending on the file name extension,
# io.read_geometrics chooses the type of file to be read
        self.gdata = io.read_geometrics(
            file, self.n_block, self.d_sensor, self.h1_sensor, self.dispo)
        self.segments = self.gdata.segments
        self.sensor1 = self.gdata.sensor1
        self.sensor2 = self.gdata.sensor2
# Check whether gradient data exist. For this, sensor2 must have data with
# non-zero variance (not all data are 0.) and dispo must be 0 (if it is 1 or 2,
# sensors are horizontally placed).
        if len(self.sensor1) == len(self.sensor2) and\
                not np.isclose(np.std(self.sensor2), 0.) and self.dispo == 0:
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor
            self.grad_data = True
        else:
            self.grad = np.array([0.0])
            self.grad_data = False
        self.x = self.gdata.x
        self.y = self.gdata.y
        self.z = np.ones_like(self.x) * self.h1_sensor
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        self.topo = np.zeros_like(self.x)
        self.n_data = self.gdata.n_data
        self.n_lines = self.gdata.n_lines
        self.sensor1_ori = np.copy(self.sensor1)
        self.sensor2_ori = np.copy(self.sensor2)
        self.grad_ori = np.copy(self.grad)
        self.time = self.gdata.time
        self.direction = 0
        self.data = self.lines()
        self.data["grad_data"] = self.grad_data
        self.data["year"] = self.gdata.year[0]
        self.data["dispo"] = self.dispo
        self.data["block"] = self.n_block
        self.data["height"] = self.h1_sensor
        self.data["line_declination"] = self.line_declination
        if self.grad_data:
            self.data["height2"] = self.h2_sensor
        self.data["d_sensor"] = self.d_sensor
        self.data["title"] = self.title
        self.unit = "nT"
        return True

    def correct_time(self, dt=None):
        """
        Correct time of recording in case there was a timing problem with the
        instrument.

        Parameters
        ----------
        dt : float, optional. Default: None
            Time to be added to instrument time in seconds.
            If None, time shift is ased for interactively. In this case, the
            obtaines tume shift is stored in configuratio file.

        Returns
        -------
        None

        """
        if not dt:
            d_time = self.comm.get_time_correction(dt)
            if not d_time:
                print("\nNo time correction applied")
                return
        for key in self.data.keys():
            if isinstance(key, str):
                break
            self.data[key]["time"] += d_time
        print(f"\nInstrument time corrected by {d_time:0.1f} seconds")
        if dt:
            with open(self.config_file, "a") as fo:
                fo.write(f"{d_time:0.3f}\n")

    def prepare_gdata(self, original_fill=False):
        """
        Copies actual (usually modified or non-Geometrics) data into class
        Geometrics for creation of a .stn or .dat geometrics file

        Parameters
        ----------
        original_fill : bool, optional. Default: False
            If True, gdata is fully initiated for non gridded data, i.e.,
            not only data arrays are updated, but also coordinates and times.

        Returns
        -------
        None.

        """
        if original_fill:
            now = datetime.now()
            h = now.hour
            m = now.minute
            s = 0.
            ye = now.year
            mo = now.month
            d = now.day
# If data have been interpolated onto a rehular grid, their structure should
# have changed (number of samples, number of lines etc.). Therefore, the full
# space gdata is reconstructed. Time are set to time of data creation.
        if self.inter_flag:
            if original_fill:
                self.gdata.year = [ye]
                self.gdata.month = [mo]
                self.gdata.day = [d]
                self.gdata.hour = [h]
                self.gdata.minute = [m]
            x = []
            y = []
            z = []
            s1 = []
            s2 = []
            line = []
            mark = []
            day = []
            month = []
            year = []
            hour = []
            minute = []
            second = []
            dx = self.x_inter[1] - self.x_inter[0]
            dy = self.y_inter[1] - self.y_inter[0]
            ye = self.gdata.year[0]
            mo = self.gdata.month[0]
            d = self.gdata.day[0]
            if self.gdata.hour[0]*60+self.gdata.minute[0] <\
                    self.gdata.hour[-1]*60+self.gdata.minute[-1]:
                h = self.gdata.hour[0]
                m = self.gdata.minute[0]-1
            else:
                h = self.gdata.hour[-1]
                m = self.gdata.minute[-1]-1
            s = 0.
            if dx < dy:
                for iy, yy in enumerate(self.y_inter):
                    h, m, s = u.next_minute(h, m, s)
                    s -= 0.1
                    for ix, xx in enumerate(self.x_inter):
                        h, m, s = u.add_time(h, m, s, 0.1)
                        if np.isnan(self.sensor1_inter[iy, ix]) or\
                                np.isnan(self.sensor2_inter[iy, ix]):
                            continue
                        x.append(xx)
                        y.append(yy)
                        z.append(self.z_inter[iy, ix])
                        s1.append(self.sensor1_inter[iy, ix])
                        if self.grad_data:
                            s2.append(self.sensor2_inter[iy, ix])
                        else:
                            s2.append(0.)
                        line.append(iy)
                        mark.append(0)
                        year.append(ye)
                        month.append(mo)
                        day.append(d)
                        hour.append(h)
                        minute.append(m)
                        second.append(s)
            else:
                for ix, xx in enumerate(self.x_inter):
                    h, m, s = u.next_minute(h, m, s)
                    s -= 0.1
                    for iy, yy in enumerate(self.y_inter):
                        h, m, s = u.add_time(h, m, s, 0.1)
                        if np.isnan(self.sensor1_inter[iy, ix]) or\
                                np.isnan(self.sensor2_inter[iy, ix]):
                            continue
                        x.append(xx)
                        y.append(yy)
                        z.append(self.z_inter[iy, ix])
                        s1.append(self.sensor1_inter[iy, ix])
                        if self.grad_data:
                            s2.append(self.sensor2_inter[iy, ix])
                        else:
                            s2.append(0.)
                        line.append(ix)
                        mark.append(0)
                        year.append(ye)
                        month.append(mo)
                        day.append(d)
                        hour.append(h)
                        minute.append(m)
                        second.append(s)
            self.gdata.x = np.array(x)
            self.gdata.y = np.array(y)
            self.gdata.z = np.array(z)
            self.gdata.sensor1 = np.array(s1)
            self.gdata.sensor2 = np.array(s2)
            self.gdata.line = np.array(line, dtype=int)
            self.gdata.mark = np.array(mark, dtype=int)
            self.gdata.year = np.array(year, dtype=int)
            self.gdata.month = np.array(month, dtype=int)
            self.gdata.day = np.array(day, dtype=int)
            self.gdata.hour = np.array(hour, dtype=int)
            self.gdata.minute = np.array(minute, dtype=int)
            self.gdata.second = np.array(second, dtype=float)
# If data are not interpolated onto a regular grid, it is supposed that their
# structure did not change, only the values may have changed due to diural
# corrections or others.
        else:
            self.gdata.sensor1 = np.array([])
            self.gdata.sensor2 = np.array([])
            if original_fill:
                self.gdata.x = np.array([])
                self.gdata.y = np.array([])
                self.gdata.line = np.array([], dtype=int)
                self.gdata.hour = np.array([], dtype=int)
                self.gdata.minute = np.array([], dtype=int)
                self.gdata.second = np.array([], dtype=int)
            for key, val in self.data.items():
                if original_fill:
                    h, m, s = u.next_minute(h, m, s)
                    hour = []
                    minute = []
                    second = []
                if isinstance(key, str):
                    break
                self.gdata.sensor1 = np.concatenate((self.gdata.sensor1,
                                                     val["s1"]))
                if self.grad_data:
                    self.gdata.sensor2 = np.concatenate((self.gdata.sensor2,
                                                         val["s2"]))
                else:
                    self.gdata.sensor2 = np.concatenate((
                        self.gdata.sensor2, np.zeros_like(val["s1"])))
                if original_fill:
                    for _ in val["x"]:
                        h, m, s = u.add_time(h, m, s, 0.1)
                        hour.append(h)
                        minute.append(m)
                        second.append(s)
                    self.gdata.hour = np.concatenate((self.gdata.hour,
                                                      np.array(hour)))
                    self.gdata.minute = np.concatenate((self.gdata.minute,
                                                        np.array(minute)))
                    self.gdata.second = np.concatenate((self.gdata.second,
                                                        np.array(second)))
                    self.gdata.x = np.concatenate((self.gdata.x, val["x"]))
                    self.gdata.y = np.concatenate((self.gdata.y, val["y"]))
                    ones = np.ones_like(val["x"], dtype=int)
                    self.gdata.line = np.concatenate((self.gdata.line,
                                                      ones*key))
                    self.gdata.year = np.ones_like(self.gdata.x, dtype=int)*ye
                    self.gdata.month = np.ones_like(self.gdata.x, dtype=int)*mo
                    self.gdata.day = np.ones_like(self.gdata.x, dtype=int)*d
                    self.gdata.mark = np.zeros_like(self.gdata.x, dtype=int)
            if original_fill:
                self.gdata.z = np.zeros_like(self.gdata.x, dtype=float)

    def write_geometrics(self, file):
        """
        Wrapper to write data in Geometrics MagMap2000 .stn format.

        Data must be interpolated onto a regular grid.

        Parameters
        ----------
        file : str
            File name where to write data.

        Returns
        -------
        None.

        """
        self.prepare_gdata()
        self.gdata.write_stn(file)

    def write_dat(self, file, original_fill=False):
        """
        Wrapper to write data in Geometrics MagMap2000 .dat format.

        Parameters
        ----------
        file : str
            File name where to write data.
        original_fill : bool, optional. Default: False
            If True, gdata is fully initiated for non gridded data, i.e.,
            not only data arrays are updated, but also coordinates and times.

        Returns
        -------
        None.

        """
        self.prepare_gdata(original_fill=original_fill)
        self.gdata.write_dat(file)

    def read_txt(self, file, height1=None, height2=None, dec=None, title=None):
        """
        Reads a non-Geometrics format magnetic data file
        This option is mainly thought for reading the output of program mgwin
        used with the option to enter all data points with their specific
        positions (NBPTS > 0). This allows calculation of a 2D map with mgwin.
        You may use Prepare_mgwin_calculation_points.py to preapre the
        coordinates.

        The file structure is as follows:

        - One comment line
        - line with NX, NY (number of points in X and Y direction)
          It is supposed that the data have been calculated on a regular
          grid. mgwin writes on this line the total number of data points
          the file must therefore be edited to replace the existing number
          by the two required ones.
        - one line per data point with (X, Y, Z, DATA)

        mgwin writes only one value into the file and in order to keep the
        structure of Geometrics simple, these values are copied into both
        sensor1 and sensor2 arrays.  Data are copied as well into
        self.sensor_n (1D array) and self.sensor_n_inter (2D array)

        Parameters
        ----------
        file : str
            Name of file to be read
        height1 : float, optional. Default: None
            height of sensor 1 above ground (meters)
        height2 : float, optional. Default: None
            height of sensor 2 above ground (meters)
        dec : float, optional. Default: None
            Direction of y axis with respect to North (magnetic north for
            magnetic data, geographic north for gravity).
        title : str, optional. Default None
            Title appearing on most plots

        Returns
        -------
        data

        """
        if height1 is None:
            ret, self.h1_sensor, self.h2_sensor, self.line_declination, \
                self.title = comm.get_geometry(
                    file, h1=self.h1_sensor, h2=self.h2_sensor,
                    dec=self.line_declination, title="Data")
            if not ret:
                print("Program aborted")
                sys.exit()
        else:
            self.h1_sensor = height1
            self.h2_sensor = height2
            self.d_sensor = height2 - height1
            self.line_declination = dec
            self.title = title
        with open(file, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        nums = lines[1].split()
        nx = int(nums[0])
        ny = int(nums[1])
        self.sensor1_inter = np.zeros((ny, nx))
        xx = np.zeros((ny, nx))
        yy = np.zeros((ny, nx))
        zz = np.zeros((ny, nx))
        ll = lines[2].split()
        if len(ll) > 4:
            grad_flag = True
            self.grad_data = True
            self.sensor2_inter = np.zeros((ny, nx))
        else:
            grad_flag = False
        n = 1
        line = -1
        for i in range(nx):
            line += 1
            self.segments[line] = {}
            n1 = n - 1
            self.segments[line]["mark_samples"] = [n1]
            self.segments[line]["dx"] = []
            self.segments[line]["dy"] = []
            self.segments[line]["d"] = []
            for j in range(ny):
                n += 1
                nums = lines[n].split()
                xx[j, i] = float(nums[0])
                yy[j, i] = float(nums[1])
                zz[j, i] = float(nums[2])
                self.x.append(float(nums[0]))
                self.y.append(float(nums[1]))
                self.sensor1_inter[j, i] = float(nums[3])
                self.sensor1.append(float(nums[3]))
                if grad_flag:
                    self.sensor2_inter[j, i] = float(nums[4])
                    self.sensor2.append(float(nums[4]))
            self.segments[line]["mark_samples"].append(n - 1)
            n2 = n - 2
            self.segments[line]["dx"].append(abs(self.x[-1] - self.x[-2]))
            self.segments[line]["dy"].append(abs(self.y[-1] - self.y[-2]))
            self.segments[line]["d"].append(
                np.sqrt(self.segments[line]["dx"][-1] ** 2
                        + self.segments[line]["dy"][-1] ** 2))
            self.segments[line]["median1"] = np.nanmedian(self.sensor1[n1:n2])
            self.segments[line]["median2"] =\
                np.nanmedian(self.sensor2_inter[n1:n2])
            self.segments[line]["x"] = np.nanmedian(self.x[n1:n2])
            self.segments[line]["y"] = np.nanmedian(self.y[n1:n2])
            self.segments[line]["mask"] = True
            self.segments[line]["block"] = self.n_block
            self.segments[line]["direction"] = self.line_declination
            self.segments[line]["dir"] = "odd"
            self.segments[line]["pos"] = self.segments[line]["x"]
            if grad_flag:
                self.segments[line]["sensor"] = 0
            else:
                self.segments[line]["sensor"] = 1
        self.sensor1 = np.array(self.sensor1)
        self.sensor2 = np.array(self.sensor2)
        self.n_data = len(self.sensor1)
        self.n_lines = nx
        if grad_flag:
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor
# Store original data to arrays xxx_ori
        self.sensor1_ori = np.copy(self.sensor1)
        self.sensor2_ori = np.copy(self.sensor2)
        self.grad_ori = np.copy(self.grad)
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.z = np.ones_like(self.x) * self.h1_sensor
        self.topo = np.zeros_like(self.x)
        self.time = np.arange(len(self.x)) * 0.1
        self.x_inter = np.unique(self.x)
        self.y_inter = np.unique(self.y)
        self.direction = 0
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        self.data = self.lines()
        self.data["grad_data"] = grad_flag
        self.data["year"] = 0
        self.data["dispo"] = 0
        self.data["block"] = self.n_block
        self.data["height"] = self.h1_sensor
        self.data["line_declination"] = self.line_declination
        if grad_flag:
            self.data["height2"] = self.h2_sensor
        self.data["d_sensor"] = self.d_sensor
        self.data["title"] = self.title
        if "m" in self.data_type:
            self.unit = "nT"
        else:
            self.unit = "mGal"
        self.prepare_gdata(original_fill=True)
        del self.segments
        # return self.data

    def read_gxf(self, infile, height=None, dec=None, title=None):
        """
        Read a gxf file (BRGM magnetic and gravity gridded files)

        Parameters
        ----------
        infile: string
            Name of file to be read
        height : float, optional. Default: None
            height of sensor 1 above ground (meters)
        dec : float, optional. Default: None
            Direction of y axis with respect to North (magnetic north for
            magnetic data, geographic north for gravity).
        title : str, optional. Default None
            Title appearing on most plots
        """
        if height is None:
            if "m" in self.data_type:
                h1 = 85.
                text = "Magnetic"
                self.unit = "nT"
            else:
                h1 = 0.
                text = "Gravity"
                self.unit = "mGal"
            ret, self.h1_sensor, self.line_declination,  self.title = \
                comm.get_geometry(infile, h1=h1, dec=self.line_declination,
                                  title=text)
            if not ret:
                print("Program aborted")
                sys.exit()
        else:
            self.h1_sensor = height
            self.line_declination = dec
            self.title = title
        with open(infile, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        # Read header
        il = -1
        n_rows = 0
        n_cols = 0
        x_origin = 0.0
        y_origin = 0.0
        dummy = 0.0
        while True:
            il += 1
            if lines[il][:5] == "#GRID":
                break
            if lines[il][:7] == "#POINTS":
                il += 1
                n_cols = int(lines[il])
            elif lines[il][:5] == "#ROWS":
                il += 1
                n_rows = int(lines[il])
            elif lines[il][:13] == "#PTSEPARATION":
                il += 1
                self.dx = float(lines[il])
            elif lines[il][:13] == "#RWSEPARATION":
                il += 1
                self.dy = float(lines[il])
            elif lines[il][:8] == "#XORIGIN":
                il += 1
                x_origin = float(lines[il])
            elif lines[il][:8] == "#YORIGIN":
                il += 1
                y_origin = float(lines[il])
            elif lines[il][:6] == "#DUMMY":
                il += 1
                dummy = float(lines[il])
        data = np.zeros((n_rows, n_cols))
        self.n_lines = n_rows
        self.grad_data = False
        self.direction = 1
        self.x_inter = x_origin + np.arange(n_cols) * self.dx
        self.y_inter = y_origin + np.arange(n_rows) * self.dy
        # Read data
        c2 = 0
        for ir in range(n_rows):
            c1 = c2
            self.segments[ir] = {}
            ic2 = 0
            while True:
                il += 1
                line = np.array(lines[il].split(), dtype=float)
                ic1 = ic2
                ic2 += len(line)
                data[ir, ic1:ic2] = line
                if ic2 == n_cols:
                    break
            data[data == dummy] = np.nan
            d = data[ir, :]
            index = np.isfinite(d)
            d = d[index]
            c2 = c1 + len(d)
            self.sensor1.extend(list(d))
            self.x.extend(list(self.x_inter[index]))
            self.y.extend(list(np.ones_like(self.x_inter[index]) *
                               self.y_inter[ir]))
            self.segments[ir]["mark_samples"] = [c1, c2]
            self.segments[ir]["dx"] = [self.dx]
            self.segments[ir]["dy"] = [0]
            self.segments[ir]["d"] = [self.dx]
            self.segments[ir]["median1"] = np.nanmedian(d)
            self.segments[ir]["median2"] = 0.
            self.segments[ir]["x"] = np.nanmedian(self.x_inter)
            self.segments[ir]["y"] = self.y_inter[ir]
            self.segments[ir]["mask"] = True
            self.segments[ir]["block"] = self.n_block
            self.segments[ir]["dir"] = "odd"
            self.segments[ir]["pos"] = self.y_inter[ir]
            self.segments[ir]["direction"] = 90.
            self.segments[ir]["sensor"] = 1
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.z = np.ones_like(self.x) * self.h1_sensor
        self.topo = np.zeros_like(self.x)
        self.sensor1 = np.array(self.sensor1)
# Store original data to arrays xxx_ori
        self.sensor1_ori = np.copy(self.sensor1)
        self.sensor1_inter = np.copy(data)
        self.time = np.arange(len(self.x))
        self.data = self.lines()
        self.data["grad_data"] = False
        self.data["year"] = 0
        self.data["dispo"] = 0
        self.data["block"] = self.n_block
        self.data["height"] = self.h1_sensor
        self.data["line_declination"] = self.line_declination
        self.data["height2"] = self.h1_sensor
        self.data["d_sensor"] = 0.
        self.data["title"] = self.title
        self.prepare_gdata(original_fill=True)
        del self.segments
        # return self.data

    def store_gxf(self, file, data, x0, y0, dx, dy):
        """
        store gridded data in GXF format

        Parameters
        ----------
        file : str
            File name.
        data : 2D numpy float array with shape (nrows, ncolumns)
            Gridded data to be stored.
        x0 : float
            Minimum X coordinate [m].
        y0 : float
            Minimum Y coordinate [m].
        dx : float
            Grid step in X direction [m].
        dy : float
            Grid step in Y direction [m].

        Returns
        -------
        None.

        """
        cols = data.shape[1]
        rows = data.shape[0]
        with open(file, "w") as fo:
            fo.write(f"#TITLE\n\n#POINTS\n{cols}\n#ROWS\n{rows}\n#SENSE\n1\n")
            fo.write(f"#XORIGIN\n{x0}\n#YORIGIN\n{y0}\n#ROTATION\n0\n")
            fo.write(f"#PTSEPARATION\n{dx}\n#RWSEPARATION\n{dy}\n#TRANSFORM\n")
            fo.write("1 0\n#UNIT_LENGTH\nm, 1\n#MAP_PROJECTION\nUTM31\n")
            fo.write("#DUMMY\n-1e32\n\n\n#GRID\n")
            for iy in range(rows):
                nc = 0
                for ix in range(cols):
                    if np.isnan(data[iy, ix]):
                        fo.write("-1e32 ")
                    else:
                        fo.write(f"{data[iy, ix]:0.3f} ")
                    nc += 1
                    if nc == 8 or ix == cols-1:
                        fo.write("\n")
                        nc = 0

    def read_BRGM_flight(self, file, title=None):
        """
        Reads magnetic data from flight lines out of a BRGM data file

        Parameters
        ----------
        file : str
            Name of file containing the data.
        title : str, optional. Default None
            Title appearing on most plots

        Returns
        -------
        x :     1D numpy float array
            E coordinate of each measured point along the line
            [Lambert 2 extended, meters]
        y :     1D numpy float array
            N coordinate of each measured point along the line
            [Lambert 2 extended, meters]
        v :     1D numpy float array
            Magnetic anomaly (measured field minus IGRF) [nT]
        topo :  1D numpy float array
            DMT topography at each measured point along the line [m]
        height : 1D numpy float array
            Flight height above topo at each measured point along the line [m]
        num : int
            Number of flight line (same as line if line < 100000.)
        """
        if not title:
            ret, self.title = comm.get_geometry(file, dx=1000, topo=True,
                                                title="Magnetic")
            if not ret:
                print("Program aborted")
                sys.exit()
        else:
            self.title = title
        self.unit = "nT"
        line_number = []
        nl = -1
        x = []
        y = []
        t = []
        height = []
        v = []
        x_line = []
        c2 = 0
        self.x = np.array([])
        with open(file, "r", encoding="utf-8") as fi:
            lines = fi.readlines()
        for _, line in enumerate(lines):
            if "/" in line:
                continue
            if "Line" in line:
                line_number.append(int(line.split()[1]))
                nl1 = nl
                nl += 1
                if nl == 0:
                    continue
                self.sensor1.extend(v)
                self.x = np.concatenate((self.x, np.array(x)))
                self.y.extend(y)
                self.z.extend(height)
                self.topo.extend(t)
                self.segments[nl1] = {}
                c1 = c2
                c2 += len(x)
                self.segments[nl1]["mark_samples"] = [c1, c2]
                self.segments[nl1]["dx"] = [1000.0]
                self.segments[nl1]["dy"] = [abs(y[-1] - y[0]) / (len(y) - 1)]
                self.segments[nl1]["d"] = [self.segments[nl1]["dy"]]
                self.segments[nl1]["median1"] = np.nanmedian(v)
                self.segments[nl1]["median2"] = 0.
                self.segments[nl1]["x"] = np.round(np.nanmedian(x), -2)
                self.x[c1:c2] = self.segments[nl1]["x"]
                x_line.append(self.segments[nl1]["x"])
                self.segments[nl1]["y"] = np.nanmedian(y)
                self.segments[nl1]["mask"] = True
                self.segments[nl1]["block"] = self.n_block
                self.segments[nl1]["dir"] = "odd"
                self.segments[nl1]["pos"] = self.segments[nl1]["x"]
                self.segments[nl1]["direction"] = 0.
                self.segments[nl1]["sensor"] = 1
                x = []
                y = []
                t = []
                height = []
                v = []
            else:
                val = line.split()
                x.append(np.round(float(val[0]), -2))
                y.append(float(val[1]))
                v.append(float(val[14]))
                t.append(float(val[6]))
                height.append(float(val[4])+t[-1])
        self.sensor1.extend(v)
        self.x = np.concatenate((self.x, x))
        self.y.extend(y)
        self.z.extend(height)
        self.topo.extend(t)
        self.n_lines = nl
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.topo = np.array(self.topo)
        self.sensor1 = np.array(self.sensor1)
        self.xmin = self.x.min()
        self.xmax = self.x.max()
        self.ymin = self.y.min()
        self.ymax = self.y.max()
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        self.grad_data = False
        self.segments[nl] = {}
        c1 = c2
        c2 += len(x)
        self.segments[nl]["mark_samples"] = [c1, c2]
        self.segments[nl]["dx"] = [1000.0]
        self.segments[nl]["dy"] = [abs(y[-1] - y[0]) / (len(y) - 1)]
        self.segments[nl]["d"] = [self.segments[nl]["dx"]]
        self.segments[nl]["median1"] = np.nanmedian(v)
        self.segments[nl]["median2"] = 0.
        self.segments[nl]["x"] = np.round(np.nanmedian(x), -2)
        self.x[c1:c2] = self.segments[nl]["x"]
        x_line.append(self.segments[nl1]["x"])
        self.segments[nl]["y"] = np.round(np.nanmedian(y), 3)
        self.segments[nl]["mask"] = True
        self.segments[nl]["block"] = self.n_block
        self.segments[nl]["dir"] = "odd"
        self.segments[nl]["pos"] = self.segments[nl]["x"]
        self.segments[nl]["direction"] = 0.
        self.segments[nl]["sensor"] = 1
        x_line = np.array(x_line)
        dx_line = np.median(abs(x_line[1:]-x_line[:-1]))
        for key, val in self.segments.items():
            val["dx"] = dx_line
    # Store original data to arrays xxx_ori
        self.sensor1_ori = np.copy(self.sensor1)
        self.time = np.arange(len(self.x))
        self.data = self.lines()
        self.data["grad_data"] = False
        self.data["year"] = 0
        self.data["dispo"] = 0
        self.data["block"] = self.n_block
        self.data["height"] = np.mean(self.z-self.topo)
        self.data["height2"] = 0.
        self.data["line_declination"] = 0.0
        self.data["d_sensor"] = 0.
        self.data["title"] = self.title
        self.prepare_gdata(original_fill=True)
        del self.segments
        self.topo = -self.topo
        self.topo_flag = True

        # return self.data

    def get_line(self, i_line=0):
        """
        Get data of one single line

        Parameters
        ----------
        i_line : int, optional. Default is 0.
            Number of line to be extracted (counting starts at 0).

        Returns
        -------
        sensor1: numpy float array
            Data of sensor 1.
        sensor2: numpy float array
            Data of sensor 2.
        x: numpy float array
            X-coordinate of all data points extracted.
        y: numpy float array
            Y-coordinate of all data points extracted.
        z: numpy float array
            height of lower sensor above ground of all data points.
        time: numpy int array
            second of day of acquisition of all data points extracted.
        mask: bool
            mask whether data should be plotted (True) or not (False)
        direction: str
            Line direction with respect to magnetic North
            (positive from N to E)
        sensor: int
            if 0, two sensors are used in vertical configuration. 1 if only one
            sensor was used or sensor 1 in horizontal configuration.
            2: sensor 2 in horizontal configuration.
        median1: float
            Median value of line for data of sensor1.
        median2: float
            Median value of line for data of sensor2.
        block: int
            Number of data set having been read

        """
        n1 = self.segments[i_line]["mark_samples"][0]
        n2 = self.segments[i_line]["mark_samples"][-1]
        if self.grad_data:
            return (self.sensor1[n1:n2], self.sensor2[n1:n2], self.x[n1:n2],
                    self.y[n1:n2], self.z[n1:n2], self.topo[n1:n2],
                    self.time[n1:n2],
                    self.segments[i_line]["mask"],
                    self.segments[i_line]["direction"],
                    self.segments[i_line]["sensor"],
                    self.segments[i_line]["median1"],
                    self.segments[i_line]["median2"],
                    self.segments[i_line]["block"])
        dum = np.array([0])
        return (self.sensor1[n1:n2], dum, self.x[n1:n2], self.y[n1:n2],
                self.z[n1:n2], self.topo[n1:n2], self.time[n1:n2],
                self.segments[i_line]["mask"],
                self.segments[i_line]["direction"],
                self.segments[i_line]["sensor"],
                self.segments[i_line]["median1"],
                self.segments[i_line]["median2"],
                self.segments[i_line]["block"])

    def lines(self):
        """
        Put all data into a simplified dictionary, one entry per line

        No input parameters

        Returns
        -------
        data: dictionary with one entry per line, key = number of line.
              Each entry is itself a dictionary containing the following
              entries:

              - "s1": Numpy float array with data of sensor 1
              - "s2": Numpy float array with data of sensor 2
                If only data of one single sensor were measured, "s2" contains
                only one zero.
              - "x":  Numpy float array with E-W coordinates of data points
              - "y":  Numpy float array with N-S coordinates of data points
              - "grad_flag" bool. True if 2 sensors were used, False if only
                one sensor
              - "mask": bool, True if line should be plotted, False if excluded
                from plotting
              - "direction": str, direction of a line with respect to magnetic
                N positive from N to E
              - "sensor":
                if 0, two sensors are used in vertical configuration. 1 if
                only one sensor was used or sensor 1 in horizontal
                configuration. 2: sensor 2 in horizontal configuration.
              - "median1": float, median of data from sensor 1
              - "median2": float, median of data from sensor 2
              - "block": int, number of data set having been read
        """
        data = {}
        for i, _ in self.segments.items():
            s1, s2, x, y, z, topo, t, m, d, s, med1, med2, block = \
                self.get_line(i)
            data[i] = {}
            data[i]["s1"] = s1
            data[i]["s2"] = s2
            data[i]["x"] = x
            data[i]["y"] = y
            data[i]["z"] = z
            data[i]["topo"] = topo
            data[i]["time"] = t
            data[i]["mask"] = m
            data[i]["direction"] = d
            data[i]["sensor"] = s
            data[i]["median1"] = med1
            data[i]["median2"] = med2
            data[i]["block"] = block
        return data

    def read_base(self, file, year):
        """
        Wrapper to read base station data

        Parameters
        ----------
        file : str
            Name of data file containing base station data.
        year : int
            Year of data acquisition (often, data contain only day, not year).

        Returns
        -------
        None.

        """
        self.base_ini()
        if "temp" not in file:
            print(f"\nRead base station file {file}")
        self.base.read_base(file, year=year)

    def base_ini(self):
        """
        Create an empty container for base station data

        Returns
        -------
        None.

        """
        self.base = Geometrics()

    def write_base(self, file):
        """
        Wrapper to write base station data. May be used if time variations are
        calculated from data or if original base station data were modified,
        normally by muting erroneous data

        Parameters
        ----------
        file : str
            Name of output file.

        Returns
        -------
        None.

        """
        self.base.write_base(file)

    def interpol(self):
        """
        Interpolate data within the measurement lines onto a regular grid.
        No data are extrapolated, i.e. if a line starts later or finishes
        earlier than a regular grid, missing grid points are set to nan

        Returns
        -------
        None.

        """
# Calculate proposal for grid steps: inline as average sample interval along
# first line, cross-line as distance between first two lines.
        if self.data[0]["direction"] in ("N", "S", 0., 180.):
            ddy = np.round(
                abs(self.data[0]["y"][-1] - self.data[0]["y"][0])
                / (len(self.data[0]["y"])-1), 2)
            ddx = np.round(abs(self.data[0]["x"][0] - self.data[1]["x"][0]), 2)
        else:
            ddx = np.round(
                abs(self.data[0]["x"][-1] - self.data[0]["x"][0])
                / (len(self.data[0]["x"])-1), 2)
            ddy = np.round(abs(self.data[0]["y"][0] - self.data[1]["y"][0]), 2)

# Open dialogue box and ask for grid steps.
# The while loop gives the opportunity to modify settings if the number of
# grid points becomes too large
        while True:
            results, okButton = dialog(
                ["dx [m]", "dy [m]", "fill holes"], ["e", "e", "c"],
                [ddx, ddy, 1], "Interpolation parameters")
            if okButton:
                self.dx = float(results[0])
                self.dy = float(results[1])
                self.hole_flag = int(results[2] > -1)
            else:
                print("\nInterpolation cancelled")
                return False

# Check size of grid. If more than one million data points are expected, give
# a warning message
            n_inter_x = int((self.xmax - self.xmin) / self.dx)
            n_inter_y = int((self.ymax - self.ymin) / self.dy)
            if n_inter_x*n_inter_y > 1000000:
                answer = QtWidgets.QMessageBox.warning(
                    None,
                    "Warning",
                    f"dx/dy={self.dx}/{self.dy} produces huge grid "
                    + f"({n_inter_x}x{n_inter_y} points)\n "
                    + "Ignore and continue or\n  close and try again\n",
                    QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Close,
                    QtWidgets.QMessageBox.Close)
# If user wand to change setting, go béck to beginning of while loop
                if answer == QtWidgets.QMessageBox.Close:
                    continue
                break
            break

# do interpolation
        self.inter_flag = True
        if self.grad_data:
            (self.sensor1_inter, self.sensor2_inter, self.grad_inter,
             self.x_inter, self.y_inter, self.z_inter, self.t_inter,
             self.topo_inter) =\
                u.interpol_2D(self, dx=self.dx, dy=self.dy,
                              fill_hole=self.hole_flag)
        else:
            self.sensor1_inter, _, _, self.x_inter, self.y_inter, \
                self.z_inter, self.t_inter, self.topo_inter = u.interpol_2D(
                    self, dx=self.dx, dy=self.dy, fill_hole=self.hole_flag)
        self.mask1 = np.isnan(self.sensor1_inter)
        self.sensor1_fill = u.extrapolate(
            self.sensor1_inter, self.x_inter, self.y_inter)
        self.z_fill = u.extrapolate(
            self.z_inter, self.x_inter, self.y_inter)
        if self.data["grad_data"]:
            self.mask2 = np.isnan(self.sensor2_inter)
            self.sensor2_fill = u.extrapolate(
                self.sensor2_inter, self.x_inter, self.y_inter)
            self.grad_fill = (self.sensor2_fill - self.sensor1_fill)\
                / self.data["d_sensor"]
        self.treatments["interpol"] = True

    def nan_fill(self):
        """
        Fill nan values by extrapolation of data in the direction perpendicular
        to the measurement direction (it is supposed that if a line is not
        complete, nearby ones will be). Extrapolation will be done (one wants
        to create a complete grid) and different possibilities exist (mainly
        spline or constant). Spline is often very risky.

        Returns
        -------
        None.

        """
        self.treatments["nan_fill"] = True
        self.nan_flag = not self.nan_flag
        if self.nan_flag:
            self.sensor1_back = np.copy(self.sensor1_inter)
            self.sensor1_inter = u.extrapolate(self.sensor1_inter,
                                               self.x_inter, self.y_inter)
            if self.grad_data:
                self.sensor2_back = np.copy(self.sensor2_inter)
                self.sensor2_inter = u.extrapolate(self.sensor2_inter,
                                                   self.x_inter, self.y_inter)
                self.grad_inter = (self.sensor2_inter - self.sensor1_inter)\
                    / self.d_sensor
            self.treatments["nan_fill"] = True
        else:
            self.sensor1_inter = np.copy(self.sensor1_back)
            if self.grad_data:
                self.sensor2_inter = np.copy(self.sensor2_back)
            self.treatments["nan_fill"] = False

    def clean(self):
        """
        Delete erroneous data
        Opions are
        * giving fixed upper and/or lower bounds (in or mGal)
        * eliminating all data below and/or above a certain quantile
        * choose limits by mouse click an a histogram (same value for both
        sensors). First click: cut below, second: cut above. a click outside
        an axis means not clipping in this direction.
        Data outside the chosen zone are set to Nan.

        Returns
        -------
        None.

        """
# Get extreme values where to clip values
        ret, min_fix, max_fix, percent_down, percent_up, histo =\
            comm.clip_parameters()
        if not ret:
            print("\nclipping cancelled")
            return
# If extreme values should be chosen manually in histogram, do this now
# Plot 1 or 2 histograms depending on the number of sensors used
        if histo is True:
            min_fix, max_fix = FP.histo_plot(self)
            if min_fix is not None:
                percent_down = None
            if max_fix is not None:
                percent_up = None
# Delet data outside limits
        u.clean_data(self, min_fix=min_fix, max_fix=max_fix,
                     percent_down=percent_down, percent_up=percent_up)
        self.treatments["clip"] = True

    def justify_median(self):
        """
        Adjust medians of every second line to the average of the medians of
        the neighboring lines in order to attenuate the directional effects
        of measurements.
        The lines to be modified may be the even ones or the odd ones (does not
        always give the same result). Medians of lines at the edges are set to
        the same value as the neighboring line.

        """
        ret, justify = comm.get_justify_indices()
        if not ret:
            print("\nJustification cancelled")
            return
        u.justify_lines_median(self, justify)
        self.treatments["justify_median"] = True

    def justify_gauss(self):
        """
        see Masoudi et al., J. Geophys. Eng., 2023

        Returns
        -------
        None.

        """
        ret, justify, local = comm.get_justify_indices(glob=True)
        if not ret:
            print("\nJustification cancelled")
            return
        u.justify_lines_gaussian(self, justify, local)
        self.treatments["justify_Gauss"] = True

    def plot_median(self):
        """
        Plot medians of every measured line (from non-interpolated data). Odd
        lines and even lines are plotted with different colours. Also sensor1
        and sensor 2 data are plotted in to the same axis and distinuished by
        different colors.

        """
        median1_even = []
        median1_odd = []
        if self.grad_data:
            median2_even = []
            median2_odd = []
        else:
            median2_even = [None]
            median2_odd = [None]
        nline_even = []
        nline_odd = []
        key1 = list(self.data.keys())[0]
        dir_odd = self.data[key1]["direction"]
        dir_even = dir_odd
        for key, val in self.data.items():
            if isinstance(key, (str)):
                break
            if val["direction"] != dir_odd:
                dir_even = val["direction"]
                break
        if dir_odd in ("N", 0.):
            txt_odd = "N"
            txt_even = "S"
        elif dir_odd in ("S", 180.):
            txt_odd = "S"
            txt_even = "N"
        elif dir_odd in ("E", 90.):
            txt_odd = "E"
            txt_even = "W"
        else:
            txt_odd = "W"
            txt_even = "E"
        for key, val in self.data.items():
            if isinstance(key, (str)):
                break
            if val["direction"] == dir_even:
                median1_even.append(val["median1"])
                if self.grad_data:
                    median2_even.append(val["median2"])
                nline_even.append(key + 1)
            else:
                median1_odd.append(val["median1"])
                if self.grad_data:
                    median2_odd.append(val["median2"])
                nline_odd.append(key + 1)
        FP.median_plot(self.data, median1_even, median1_odd, median2_even,
                       median2_odd, nline_even, nline_odd, txt_even, txt_odd)

    def reduce_pole(self):
        """
        Pole reduction is done, only for the external field, eventual remanent
        magnetization is not taken into account.

        Returns
        -------
        None.

        """
        self.treatments["pole"] = True
        self.sensor1_inter = trans.pole_reduction(
            self.sensor1_fill, self.dx, self.dy, self.inclination,
            self.declination)
        self.sensor1_fill = np.copy(self.sensor1_inter)
        self.sensor1_inter[self.mask1] = np.nan
        if self.grad_data:
            self.sensor2_inter = trans.pole_reduction(
                self.sensor2_fill, self.dx, self.dy, self.inclination,
                self.declination)
            self.sensor2_fill = np.copy(self.sensor2_inter)
            self.sensor2_inter[self.mask2] = np.nan
            self.grad_inter = (self.sensor2_inter - self.sensor1_inter)\
                / self.d_sensor
            self.grad_fill = (self.sensor1_fill - self.sensor2_fill)\
                / self.d_sensor
        self.inclination = 90.0
        self.declination = 0.0
        self.earth.inc = 90.
        self.earth.earth_components()

    def spector(self):
        """
        Calculate depth of random sources with formula of (Spector and Grant,
        Geophysics, 1970) for all lines (N-S or E-W direction).
        Depths are calculated by fitting two lines to logarithmic spectrum. The
        break point between the two lines is searched between the 4th and the
        10th spectral coefficient.
        Results of all lines are saved in file spector.dat.

        """
# Check whether data are interpolated
        if not self.inter_flag:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "Data are not yet interpolated.\nYou should do this "
                + "before calling FFT.",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return
# Wait for mouse click to choose line for which the result will be shown on
#   screen
        max_len_x = self.x_inter.max() - self.x_inter.min()
        max_len_y = self.y_inter.max() - self.y_inter.min()
        max_len = [max_len_y, max_len_x]
        if self.data[0]["direction"] in ("N", "S", 0., 180.):
            direction = 0
        else:
            direction = 1
        ret, direction, half_width = comm.get_spector1D(direction, max_len)
        if not ret:
            print("No spectral analysis done")
            return
        lpos, depths1, depths2, intercepts1, intercepts2, isplits, fits, \
            n_Ny, dsamp = trans.spector1D(self, direction, half_width)
# Pot results and allow manual modification
        depths1, intercepts1 = FP.spector1D_plot(
            self, depths1, intercepts1, depths2, intercepts2, direction,
            half_width)
# Store all calculated depths into file "spector.dat
        file = u.file_name("spector", f"dat_{self.data['type']}")
        with open(file, "w", encoding="utf-8") as fo:
            fo.write("Line nr  position [m]    depth1 [m]    depth2 [m]   "
                     + " misfit\n")
            for i, lp in enumerate(lpos):
                fo.write(f"{i:7d}{lp:10.2f}{depths1[i]:10.2f}"
                         + f"{depths2[i]:10.2f}    {fits[i]:0.5f}\n")

    def spector2D(self):
        """
        Calculate source depths via spectral analysis in two dimensions

        Returns
        -------
        None.

        """
        if not self.inter_flag:
            _ = QtWidgets.QMessageBox.warning(
                None, "Warning",
                "Data are not yet interpolated.\n"
                + "You should do this before calling FFT.",
                QtWidgets.QMessageBox.Close, QtWidgets.QMessageBox.Close)
            return False

        ok, xpos, ypos, depths1, depths2, intercepts1, intercepts2, fits, \
            window_len, nwiny, nwinx, step, half_width, n_Ny = \
            trans.spector2D(self)
        if not ok:
            return
        depths1, intercepts1 = FP.spector2D_plot(
            self, depths1, intercepts1, depths2, intercepts2, xpos, ypos,
            half_width, n_Ny, window_len, nwiny, nwinx, "rainbow")

        with open(f"spector2D_{self.data_type}.dat",
                  "w", encoding="utf-8") as fo:
            fo.write("Line nr  X [m]      Y[m]    depth1 [m]    depth2 [m]    "
                     + "misfit\n")
            nr, nc = depths1.shape
            for i in range(nc):
                for j in range(nr):
                    if np.isfinite(depths1[j, i]):
                        fo.write(
                            f"{i:7d}{xpos[i]:10.1f}{ypos[j]:10.1f}"
                            + f"{depths1[j,i]:10.2f}{depths2[j,i]:10.2f}"
                            + f"    {fits[j,i]:0.5f}\n")
        return True

    def tilt(self):
        """
        Calculate tilt angle (Miller & Singh, JAG, 1994)

        Returns
        -------
        None.

        """
        if self.data["grad_data"]:
            self.tilt_ang, self.vgrad, self.vgrad2, self.hgrad = trans.tilt(
                self.sensor1_fill, self.dx, self.dy, self.grad_fill)
        else:
            self.tilt_ang, self.vgrad, self.vgrad2, self.hgrad = trans.tilt(
                self.sensor1_fill, self.dx, self.dy, None)
        self.tilt_grd = trans.gradient(self.tilt_ang, self.dx, self.dy)*1000.
        max_pos, _ = u.min_max2D(self.tilt_grd, half_width=3)
        self.tilt_grd[:, :] = 0.
        self.tilt_grd[max_pos[0], max_pos[1]] = 1.
        self.tilt_ang[self.mask1] = np.nan
        self.tilt_grd[self.mask1] = np.nan
        self.vgrad[self.mask1] = np.nan
        self.vgrad2[self.mask1] = np.nan
        self.hgrad[self.mask1] = np.nan
        fig_grad = FP.plot_gradients(self, color="rainbow")

        FP.plot_tilt(self, color="rainbow")
        fig_grad.close()
        if self.lineaments is not None:
            self.nlineaments = len(self.lineaments)
        else:
            self.nlineaments = 0
        if self.nlineaments > 0:
            with open("tilt.dat", "w", encoding="utf-8") as fo:
                fo.write("    X       Y     angle\n")
                for i, x in enumerate(self.x_inter):
                    for j, y in enumerate(self.y_inter):
                        fo.write(f"{x:0.2f} {y:0.2f} "
                                 + f"{self.tilt_ang[j,i]:0.3f}\n")
        return self.lineaments

    def analytic_signal(self):
        """
        Calculate analytic signal (Nabighian, Geophysics, 1972)

        Returns
        -------
        None.

        """
# Calculate analytic signal of first sensor
        d = self.data[0]["direction"]
        if d in ("N", "S", 0., 180.):
            direction = 0
        else:
            direction = 1
        self.analytic_signal1 = trans.analytic_signal(self.sensor1_fill,
                                                      self.dx, self.dy)
# Calculate instataneous phase and frequency
        self.inst_phase1 = np.unwrap(np.angle(self.analytic_signal1))
        if direction:
            add = self.inst_phase1[-1, :].reshape(1, -1)
            self.inst_freq1 = np.diff(self.inst_phase1, append=add, axis=0)
        else:
            add = self.inst_phase1[:, -1].reshape(-1, 1)
            self.inst_freq1 = np.diff(self.inst_phase1, append=add)
# Set areas without measured data to nan
        self.analytic_signal1[self.mask1] = np.nan
        self.inst_phase1[self.mask1] = np.nan
        self.inst_freq1[self.mask1] = np.nan
# The following commented lines allow computation of instantaneous phases and
#     frequencies of the analytic signal. Since the results were not
#     convincing, I commented the plotting of these data, but maybe with other
#     data it would be worthwhile looking at them again. In this case, indent
#     the first line following the commented "else"
#
# N.B. :  Meanwhile, the program has been strongly modified. Certainly, this
# part si no longer compatible in its actual state and some reprogramming will
# have to be done.
        # if self.grad_data:
        #     analytic_signal2 = trans.analytic_signal(self.sensor2_fill,
        #                                              self.dx, self.dy)
        #     inst_phase2 = np.unwrap(np.angle(analytic_signal2))
        #     if self.w.direction:
        #         add = inst_phase2[-1,:].reshape(1,-1)
        #         inst_freq2 = np.diff(inst_phase2, append=add)
        #     else:
        #         add = inst_phase2[:,-1].reshape(-1,1)
        #         inst_freq2 = np.diff(inst_phase2, append=add)
        #     analytic_signal2[self.mask2==True] = np.nan
        #     inst_phase2[self.mask2==True] = np.nan
        #     inst_freq2[self.mask2==True] = np.nan
        # nr = analytic_signal1.shape[0]
        # nc = analytic_signal1.shape[1]
        # xmin = np.nanmin(self.x_inter)
        # xmax = np.nanmax(self.x_inter)
        # ymin = np.nanmin(self.y_inter)
        # ymax = np.nanmax(self.y_inter)
        # if self.grad_data:
        #     data = abs(analytic_signal1).reshape(nr,nc,1)
        #     data = np.concatenate(
        #         (data, abs(analytic_signal2).reshape(nr, nc, 1)), axis=2)
        #     self.fig_ana, self.ax_ana = self.w.plotFloating(
        #         data, self.w.x_inter,
        #         self.w.y_inter, wtitle="Analytic signal", sizeh=1200,
        #         sizev=900, ptitle=["Analytic signal of sensor 1",
        #                            "Analytic signal of sensor 2"],
        #         xlabel=["Easting [m]", "Easting [m]"],
        #         ylabel=["Northing [m]", "Northing [m]"],
        #         clabel=[f"Analytic signal [{self.unit}/m]",
        #                 f"Analytic signal [{self.unit}/m]"], percent=0.005)
        #     if self.w.plotLin_flag:
        #         self.w.plot_lineaments(self.ax_ana[0])
        #         self.w.plot_lineaments(self.ax_ana[1])
        #     self.ax_ana[0].set_xlim([xmin,xmax])
        #     self.ax_ana[0].set_ylim([ymin,ymax])
        #     self.ax_ana[1].set_xlim([xmin,xmax])
        #     self.ax_ana[1].set_ylim([ymin,ymax])
        # else:
        # data = abs(self.analytic_signal1)
# Plot analytic signal
        FP.plot_analytic(self, color="rainbow")

    def continuation(self):
        """
        Calculate field at higher or lower altitude
        (Nabighian, Geophysics, 1972)

        Parameters are given interactively

        Returns
        -------
        dz : float
            Distance by which to continue filed (positive = upward)
        """
        results, okButton = dialog(["Continuation_height [m] (>0: upward)"],
                                   ["e"], [10.0], "Field continuation")
        if okButton:
            self.dz = float(results[0])
        else:
            print("\nUpward continuation cancelled")
            return
        pro_data1 = trans.continuation(self.sensor1_fill, self.dx, self.dy,
                                       self.dz)
        self.sensor1_fill = np.copy(pro_data1)
        pro_data1[self.mask1] = np.nan
        self.sensor1_inter = np.copy(pro_data1)
        self.data["height"] += self.dz
        self.h1_sensor += self.dz
        if self.grad_data:
            pro_data2 = trans.continuation(self.sensor2_fill, self.dx, self.dy,
                                           self.dz)
            self.sensor2_fill = np.copy(pro_data2)
            pro_data2[self.mask1] = np.nan
            pro_datag = (pro_data1 - pro_data2) / self.d_sensor
            self.grad_fill = (self.sensor1_fill - self.sensor2_fill)\
                / self.d_sensor
            self.sensor2_inter = np.copy(pro_data2)
            self.grad_inter = np.copy(pro_datag)
            self.data["height2"] += self.dz
            self.h2_sensor += self.dz
        self.treatments["up"] = True
        return self.dz

    def delete_data(self, pos, s1, s2, x0, x1, sensor_flag, lin, line_keys,
                    index_sort, direction):
        """
        Set all data in a vector to nan that are located between coordinates
        ]x0,x1[
        Elimination is done in situ

        Parameters
        ----------
        pos : Numpy 1D float array
            Contains the coordinates of all points along the line
        s1 : Numpy 1D float array
            Data of sensor 1
        s2 : Numpy 1D float array
            Data of sensor 2
        Only one of the two, s1 or s2, is really used
        x0, x1 : floats
            Data at positions x0 < pos < x1 are set to nan
            x0 and x1 are not necessarily in ascending order, the program
            checks the order
        sensor_flag : int
            Number of sensor to be muted (1 or 2)
        lin : int
            Number of row or column for gridded data to be muted.
            Not used for original data
        line_keys : list of int
            Dictionary keys of all pertial lines integrated into the extracted
            data.
            Only used for non-gridded data
        index_sort : list of int
            Data ordering in ascending position along the line
            This list may be important if data were measured in different
            blocks and are not necessarily stored consecutively in the data
            file.
        direction : str
            Direction of line. May be "N" or "E"

        Returns
        -------
        None

        """
        index = np.where((pos > min(x0, x1)) & (pos < max(x0, x1)))
        if sensor_flag == 1:
            s1[index] = np.nan
            s1_new = np.copy(s1)
            s1_new[index_sort] = s1
            if self.inter_flag:
                if direction in ("N", "S", 0., 180.):
                    self.sensor1_inter[:, lin] = s1_new
                else:
                    self.sensor1_inter[lin, :] = s1_new
            else:
                i1 = 0
                for line in line_keys:
                    i2 = i1 + len(self.data[line]["x"])
                    self.data[line]["s1"] = np.copy(s1_new[i1:i2])
                    i1 = i2
        else:
            s2[index] = np.nan
            s2_new = np.copy(s2)
            s2_new[index_sort] = s2
            if self.inter_flag:
                if direction in ("N", "S", 0., 180.):
                    self.sensor2_inter[:, lin] = s2_new
                else:
                    self.sensor2_inter[lin, :] = s2_new
            else:
                i1 = 0
                for line in line_keys:
                    i2 = i1 + len(self.data[line]["x"])
                    self.data[line]["s2"] = np.copy(s2_new[i1:i2])
                    i1 = i2

    def get_linedata(self, xm, ym, button):
        """
        Get data from line nearest to position (xm, ym)
        If a gridded data set exists, use those data, if not use original data
        For gridded data, "button" allows to choose between extraciton the
        line in Y (button==1) or X direction (others). This is not the case
        for original data, since their line direction is given (field data).

        Parameters
        ----------
        xm : float
            X position of the cursor when pressed in data coordinates
        ym : float
            Y position of the cursor when pressed in data coordinates
        button : int
            1: left button was pressed (line in Y direction)
            3: Right button was pressed (line in X direction)

        Returns
        -------
        pos : Numpy 1D float array
            Contains the coordinates of all points along the line (Y coordinate
            for line in Y direction , X in X direction)
        pos_line : float
            Median coordinate of the line
        s1 : Numpy 1D float array
            Values of all points along the line measured with sensor 1
        s2 : Numpy 1D float array
            Values of all points along the line measured with sensor 2
            np.array([None]) if no second sensor exists
        direction : str
            May be "N" or "E"
        index_sort : list of int
            Data ordering in ascending position along the line
            This list may be important if data were measured in different
            blocks and are not necessarily stored consecutively in the data
            file. This ordering will the be used if data are set to nan
            lateron. It has no meaning for gridded data.
        line_keys : list of int
            Dictionary keys of all pertial lines integrated into the extracted
            data. Again, this is important for non-gridded data that were
            measured in different blocks and a full ine may then be composed
            of various partial lines, one from each field block.
        line_positions : Numpy 1D float array
            Median coordinates of all lines
        """
# If left mouse click, plot line in Y direction
        if self.inter_flag:
            line_keys = [None]
            if button == 1:
                lin = np.argmin(np.abs(self.x_inter-xm))
                s1 = self.sensor1_inter[:, lin]
                topo_line = self.topo_inter[:, lin]
                z_line1 = self.z_inter[:, lin]
                pos = self.y_inter
                line_positions = self.x_inter
                index_sort = np.argsort(pos)
                s1 = s1[index_sort]
                topo_line = topo_line[index_sort]
                z_line1 = z_line1[index_sort]
                if self.grad_data:
                    s2 = self.sensor2_inter[:, lin]
                    s2 = s2[index_sort]
                    z_line2 = z_line1-self.data["height"]+self.data["height2"]
                else:
                    s2 = np.array([None])
                    z_line2 = np.copy(z_line1)
                pos = pos[index_sort]
                pos_line = self.x_inter[lin]
                direction = "N"
# If right mouse click or wheel, plot line in X direction
            else:
                lin = np.argmin(np.abs(self.y_inter-ym))
                s1 = self.sensor1_inter[lin, :]
                topo_line = self.topo_inter[lin, :]
                z_line1 = self.z_inter[lin, :]
                pos = self.x_inter
                line_positions = self.y_inter
                index_sort = np.argsort(pos)
                s1 = s1[index_sort]
                topo_line = topo_line[index_sort]
                z_line1 = z_line1[index_sort]
                if self.grad_data:
                    s2 = self.sensor2_inter[lin, :]
                    s2 = s2[index_sort]
                else:
                    s2 = np.array([None])
                pos = pos[index_sort]
                pos_line = self.y_inter[lin]
                direction = "E"
            # if direction in ("N", "S", 0., 180.):
            #     if self.topo_flag:
            #         topo_line = self.topo_interpol(
            #             (np.ones_like(pos)*pos_line, pos))
            #         z_line1 = topo_line - self.z_interpol(
            #             (np.ones_like(pos)*pos_line, pos))
            #     else:
            #         topo_line = np.zeros_like(pos)
            #         z_line1 = topo_line-self.data["height"]
            #         if self.grad_data:
            #             z_line2 = topo_line-self.data["height2"]
            # else:
            #     if self.topo_flag:
            #         topo_line = self.topo_interpol(
            #             (pos, np.ones_like(pos)*pos_line))
            #         z_line1 = topo_line - self.z_interpol(
            #             (pos, np.ones_like(pos)*pos_line))
            #     else:
            #         topo_line = np.zeros_like(pos)
            #         z_line1 = topo_line-self.data["height"]
            #         if self.grad_data:
            #             z_line1 = topo_line-self.data["height2"]
# Do the same if original data are to be used
        else:
            key0 = list(self.data.keys())[0]
            direction = self.data[key0]["direction"]
            pos_l = []
            for k, val in self.data.items():
                if isinstance(k, str):
                    break
                if val["direction"] in ("N", "S", 0., 180.):
                    pos_l.append(np.nanmedian(val["x"]))
                else:
                    pos_l.append(np.nanmedian(val["y"]))
            line_positions = np.unique(np.array(pos_l))
            pos_l = np.array(pos_l)
            if self.data[key0]["direction"] in ("N", "S", 0., 180.):
                lin = np.argmin(abs(line_positions - xm))
            else:
                lin = np.argmin(abs(line_positions - ym))
            pos_line = line_positions[lin]
            line_keys = []
            pos = []
            topo_line = []
            z_line1 = []
            s1 = []
            if self.grad_data:
                s2 = []
            for k, val in self.data.items():
                if isinstance(k, str):
                    break
                if val["direction"] in ("N", "S", 0., 180.):
                    if np.isclose(np.nanmedian(val["x"]),
                                  pos_line):
                        pos += list(val["y"])
                    else:
                        continue
                else:
                    if np.isclose(np.median(val["y"]),
                                  pos_line):
                        pos += list(val["x"])
                    else:
                        continue
                s1 += list(val["s1"])
                topo_line += list(val["topo"])
                z_line1 += list(val["z"])
                if self.grad_data:
                    s2 += list(val["s2"])
                    # z_line2 += list(val("z"))
                line_keys.append(k)
            index_sort = np.argsort(pos)
            s1 = np.array(s1)
            s1 = s1[index_sort]
            topo_line = np.array(topo_line)
            topo_line = topo_line[index_sort]
            z_line1 = np.array(z_line1)
            z_line1 = z_line1[index_sort]
            if self.grad_data:
                s2 = np.array(s2)
                s2 = s2[index_sort]
                # z_line2 = np.array(z_line2)
                # z_line2 = z_line2[index_sort]
            else:
                s2 = np.array([None])
                # z_line2 = np.array([None])
            pos = np.array(pos)[index_sort]
            # if self.topo_flag:
            #     topo_line = self.topo(
            #         (np.ones_like(pos)*pos_line, pos))
            #     z_line1 = topo_line - self.z_interpol(
            #         (np.ones_like(pos)*pos_line, pos))
            # else:
            #     topo_line = np.zeros_like(pos)
            #     z_line1 = topo_line-self.data["height"]
            #     if self.grad_data:
            #         z_line2 = topo_line-self.data["height2"]
        if self.grad_data:
            z_line2 = z_line1-self.data["height2"]\
                + self.data["height"]
        else:
            z_line2 = np.copy(z_line1)
        return pos, pos_line, topo_line, z_line1, z_line2, s1, s2, lin, \
            direction, index_sort, line_keys, line_positions

    def plot_line(self, plot_flag, event0):
        """
        Plot a single line. Depending on the button pressed in the main
        window, line is plotted in Y or in X direction.

        The lines may be changed interactively by clicking the right or left
        mouse button or right and left keyboard arrow (right: increase
        coordinate, left: decrease). In this way, it is possible to show one
        line after the other.

        By pressiong "d" of "DELETE" buttons, it is possible to choose two
        points between which the data are set to nan (to eliminate mearurement
        errors). Elimination is done independently for both sensors. Modified
        data sets are automatically stored in Class data.

        If plot_flag == False, no data are plotted, no modifications are
        possible, only the data along the chosen line and their coordinates
        are exported

        Parameters
        ----------
        plot_flag : bool
            If False, only the measurement values and coordinates along the
            line are returned, no plot is done.
        event0 : QT mouse-click event
            Mouse click from the main window indicating the first line to be
            plotted and the direction (left mouse button: Y direction, right
            mouse button: X direction)

        Returns
        -------
        pos : Numpy 1D float array
            Contains the coordinates of all points along the line (Y coordinate
            for line in Y direction , X in X direction)
        pos_line : float
            Median coordinate of the line
        s1 : Numpy 1D float array
            Values of all points along the line measured with sensor 1
        s2 : Numpy 1D float array
            Values of all points along the line measured with sensor 2
            np.array([None]) if no second sensor exists
        direction : str
            May be "N" or "E"
        """
        xm = event0.xdata
        ym = event0.ydata
        button = event0.button
        fig_line_flag = False
# Get data along the chosen line
        while True:
            pos, pos_line, topo_line, z_line1, z_line2, s1, s2, lin, \
                direction, index_sort, line_keys, line_positions = \
                self.get_linedata(xm, ym, button)
            lmax = len(line_positions)-1
# If plot_flag is False, return line data
            if not plot_flag:
                return pos, pos_line, topo_line, z_line1, z_line2, s1, s2, \
                    direction
# If read_flag is True, plot the line and wait for keyboard or mouse event
# Create figure for line plot in floating window
            if not fig_line_flag:
                self.fig_line = newWindow("Single line", 1500, 1100)
                # fig_line_flag = True
# If 2 sensors have been read in, create two subplots, if not, only one.
            if self.grad_data:
                self.ax_line = self.fig_line.fig.subplots(2, 1)
                ax = None
            else:
                ax = self.fig_line.fig.subplots(1, 1)
# For simpler programming in the following part, copy single axis into a list
                self.ax_line = [ax]
# Plot sensor1 values into first axis
            self.ax_line[0].plot(pos, s1)
            self.ax_line[0].set_title(
                f'{self.title}:\nLine at {pos_line:0.2f}')
            if self.data["type"] == "magnetic":
                if self.grad_data:
                    self.ax_line[0].set_ylabel("Magnetic field [nT]\nsensor 1")
                else:
                    self.ax_line[0].set_ylabel("Magnetic field [nT]")
            else:
                if self.grad_data:
                    self.ax_line[0].set_ylabel(
                        "Gravity field [mGal]\nsensor 1")
                else:
                    self.ax_line[0].set_ylabel("Gravity field [mGal]")
# if two sensors were used, plot values of second sensor into second axis
            if self.grad_data:
                self.ax_line[1].plot(pos, s2)
                self.ax_line[1].set_xlabel("Distance [m]")
                if self.data["type"] == "magnetic":
                    self.ax_line[1].set_ylabel("Magnetic field [nT]\nsensor 2")
                else:
                    self.ax_line[1].set_ylabel(
                        "Gravity field [mGal]\nsensor 2")
            self.fig_line.setHelp(
                "Click left/right or press keypoard arrows for next line "
                + "to left/right, move mouse to point and press DEL or d to "
                + "erase data (2 points, start&end), press ENTER to stop "
                + "(move mouse a little before)")
            self.fig_line.show()
# Wait for mouse click in order to determine the next step:
#      Right click: plot next line to the East or North
#      Left click: plot next line to the West or South
            event = self.fig_line.get_event()
            next_flag = 0
# Catch keyboard press event
            if event.name == "key_press_event":
                if event.key == "enter":
                    self.fig_line.close()
                    del self.fig_line
                    return pos, pos_line, topo_line, z_line1, z_line2, s1, \
                        s2, direction
                if event.key == "right":
                    next_flag = 1
                elif event.key == "left":
                    next_flag = -1
                elif event.key == "delete" or event.key.lower == "d":
                    x0 = event.xdata
                    while True:
                        event = self.fig_line.get_event()
                        if event.name == "key_press_event":
                            if event.key == "delete" or event.key.lower == "d":
                                x1 = event.xdata
                                break
                    sensor_flag = 1
                    if "sensor 2" in event.inaxes.get_ylabel():
                        sensor_flag = 2
                    self.delete_data(pos, s1, s2, x0, x1, sensor_flag, lin,
                                     line_keys, index_sort, direction)
# Data between the two mouse clicks are muted.
# If the mouse click was detected at y > 200, mouse was in the upper subplot
#    and data of sensor 1 are muted. Else data of sensor 2 are muted.
                    index = np.where((pos > min(x0, x1)) & (pos < max(x0, x1)))
                    sensor_flag = 1
                    if "sensor 2" in event.inaxes.get_ylabel():
                        sensor_flag = 2
                    if sensor_flag == 1:
                        s1[index] = np.nan
                        s1_new = np.copy(s1)
                        s1_new[index_sort] = s1
                        if self.inter_flag:
                            if direction in ("N", "S", 0., 180.):
                                self.sensor1_inter[:, lin] = s1_new
                            else:
                                self.sensor1_inter[lin, :] = s1_new
                        else:
                            i1 = 0
                            for line in line_keys:
                                i2 = i1 + len(self.data[line]["x"])
                                self.data[line]["s1"] = np.copy(s1_new[i1:i2])
                                i1 = i2
                    else:
                        s2[index] = np.nan
                        s2_new = np.copy(s2)
                        s2_new[index_sort] = s2
                        if self.inter_flag:
                            if direction in ("N", "S", 0., 180.):
                                self.sensor2_inter[:, lin] = s2_new
                            else:
                                self.sensor2_inter[lin, :] = s2_new
                        else:
                            i1 = 0
                            for line in line_keys:
                                i2 = i1 + len(self.data[line]["x"])
                                self.data[line]["s2"] = np.copy(s2_new[i1:i2])
                                i1 = i2
                    self.fig_line.close()
                else:
                    continue
# Click was outside axes
            elif event.name == "button_press_event":
                if not event.inaxes:
                    continue
# Left click inside an axis
                if event.button == 1:
                    next_flag = -1
# Right click inside an axis
                elif event.button == 3:
                    next_flag = 1
# Wheel was clicked. The click is interpreted as one of the limits of the
#       zone to be muted
                else:
                    continue
            if next_flag < 0:
                lin = max(0, lin - 1)
                if direction in ("N", "S", 0., 180.):
                    xm = line_positions[lin]
                else:
                    ym = line_positions[lin]
            elif next_flag > 0:
                lin = min(lmax, lin + 1)
                if direction in ("N", "S", 0., 180.):
                    xm = line_positions[lin]
                else:
                    ym = line_positions[lin]
            # self.fig_line.fig.clf()
            # del self.ax_line, ax
            self.fig_line.close()
            del self.fig_line
            gc.collect()
        # self.fig_line.close()
        # del self.fig_line
        # gc.collect()
