# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 2023, 15:12:59
Last modified on Mon Oct 30 2023

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         University Paris-Saclay, France

Contains functions for data input/output

Contains functions:
    get_files
    read_geography_file
    get_mag_field

Contains class data with the following functions:
    __init__
    read_geometrics
    read_txt
    read_gxf
    read_BRGM_flight
    get_line
    lines
    write_dat

    clean_data
    line_days
    get_segment
    julian2date
    date2julian
    read_base
    write_base
    diurnal_correction
    diurnal_variation
    interpol_line
    interpol_2D
    extrapolate
#        fill_nans (actually not used)
#            eliminate_nans (actually not used)
    justify_lines_median
    justify_lines_gaussian
    matrixExtension
    poleReduction
    plot_geography
    plot_triang
    plot_image
    addMPL
    rmMPL

"""

import sys
import os
import numpy as np
from PyQt5 import QtWidgets

# from ..plotting import plot as w
from .dialog import dialog
from .geometrics import Geometrics


def get_files(dir0=None, ftype=None):
    """
    Ask for files with ending "stn".
    Several files may be chosen at once using as usual SHFT or CTRL.
    If a folder was chosen, it is automatically recognized and eliminated
    from the list of file names.

    Returns
    -------
    data_files: list str
        list of chosen files

    """
    valid_extensions = np.array([".STN", ".OUT", ".XYZ", ".GXF", ".DAT"])
    ftypes = ("GEOMETRICS", "MGWIN", "BRGM", "GXF", "OTHER")
    dtypes = ("magnetic", "gravity")
    #    sensor2_types = ["GEOMETRICS", "MGWIN"]
    try:
        os.chdir(dir0)
    except (FileNotFoundError, TypeError):
        dir0 = None
    files = list(
        QtWidgets.QFileDialog.getOpenFileNames(
            None,
            "Select data files",
            "",
            filter="stn/gxf/XYZ/dat (*.stn *.gxf *.XYZ *.dat) ;; all (*.*)",
        )
    )
    if len(files) == 0:
        print("No file chosen, program finishes")
        sys.exit("No file chosen")
    elif len(files[0]) == 0:
        print(
            "\nNo file chosen, program finishes\n\n"
            + "You probably must close the Spyder console before restarting"
        )
        sys.exit("No file chosen")
    # Sort chosen file names
    files[0].sort()
    # Check data formats
    # Set working folder as folder of the first selected file
    if ftype != "base":
        dir0 = os.path.dirname(files[0][0])
        os.chdir(dir0)
        print(f"Set working folder to: {dir0}")

    # Loop over file names and store valid file names into list data_files
    data_files = []
    file_types = []
    for _, f in enumerate(files[0]):
        _, file_ext = os.path.splitext(f)
        file_ext = file_ext.upper()
        if os.path.isdir(f):
            continue
        if file_ext not in valid_extensions:
            continue
        file_types.append(ftypes[np.where(valid_extensions == file_ext)[0][0]])
        data_files.append(f)
    if len(data_files) == 0:
        _ = QtWidgets.QMessageBox.critical(
            None,
            "Error",
            "No valid data files given\n\n"
            + f"Only {valid_extensions} allowed.\n\nProgram stops",
            QtWidgets.QMessageBox.Ok,
        )
        raise Exception("File type error.\n")
    # Ask for data types
    labels = []
    values = []
    types = []
    if ftype == "base":
        data_types = []
        for _ in range(len(files[0])):
            data_types.append("magnetic")
    else:
        for f in data_files:
            labels.append(f"{os.path.basename(f)}:")
            values.append(None)
            types.append("l")
            labels.append(["Magnetic", "Gravity"])
            values.append("0")
            types.append("r")
        results, ok_button = dialog(labels, types, values, title="data types")
        if not ok_button:
            print("No entry, program finished")
            sys.exit()
        data_types = []
        for i in range(1, len(results), 2):
            data_types.append(dtypes[int(results[i])])
    return data_files, file_types, data_types, dir0


def read_geography_file(file):
    """
    Reads file with geography information to be plotted (borders - mainly geological, towns)

    Parameters
    ----------
    file : str, name of file to be read
           File has the following structure:
               keyword may be "#POINT", "#LINE" or "#END"
               if keyword == "#POINT", one line follows with x, y coordinates and text
                                       text being the description of the point (no blanks)
               if keyword == "#LINE", several lines follow, each one with x and y
                                      coordinate of one point describing the line
               if keyword == "#END", this line finishes the data entry, possible following
                                     lines will be ignored.

    Returns
    -------
    geography : dictionary with all geography information.
                key is consecutive numbering of entries
                Each entry consists of a dictionary with the following entries:
                    "type" str
                           may be "POINT" or "LINE"
                    If type == "POINT" :
                        "x" : float
                            x coordinate of point (East)
                        "y" : float
                            y coordinate of point (North)
                        "name" : str
                                 Text to be plotted beside the point
                                 mostly name of a town
                    If type == "line" :
                        "x" : list of floats
                            x (East) coordinates of points describing the line
                        "y" : list of floats
                            y (North) coordinate of point s describing the line

    """
    with open(file, "r") as fi:
        ll = fi.readlines()
    geography = {}
    il = 0
    iunit = -1
    while True:
        if ll[il][:6].upper() == "#POINT":
            iunit += 1
            il += 1
            nums = ll[il].split()
            geography[iunit] = {}
            geography[iunit]["type"] = "POINT"
            geography[iunit]["x"] = float(nums[0])
            geography[iunit]["y"] = float(nums[1])
            geography[iunit]["name"] = nums[2]
            il += 1
        elif ll[il][:5].upper() == "#LINE":
            iunit += 1
            geography[iunit] = {}
            geography[iunit]["type"] = "LINE"
            geography[iunit]["x"] = []
            geography[iunit]["y"] = []
            while True:
                il += 1
                if ll[il][0] == "#":
                    break
                nums = ll[il].split()
                geography[iunit]["x"].append(float(nums[0]))
                geography[iunit]["y"].append(float(nums[1]))
        elif ll[il][:4].upper() == "#END":
            break
    return geography


def get_mag_field():
    """
    Get parameters of Earth's magnetic field in the study area

    Returns
    -------

    inclination : float
        Inclination of Earth's field in degrees
    declination : float
        Declination of Earth's field in degrees
    """
    results, _ = dialog(
        ["Field inclination", "Field declination"],
        ["e", "e"],
        [62, 0],
        "Earth's field parameters",
    )
    return float(results[0]), float(results[1])


class Data:
    """
    Class contains methods for data management in program PyMaGra
    """

    def __init__(self):
        print(f"Invoking __init__ from {__name__}")
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
        self.grad_data = False
        self.d_sensor = 0.9
        self.h_sensor = 0.4
        self.line_declination = 0.0
        self.interpol = 0.2
        self.data = {}
        self.n_blocks = 0
        self.n_lines = 0
        self.n_data = 0
        self.direction = 0
        self.dx = 0.0
        self.dy = 0.0
        self.line_choice = "all"

    def read_geometrics(self, file):
        """
        Read Geometrics *.stn file (G-858 instrument)

        Parameters
        ----------
        file : str
            Name of data file.

        Returns
        -------
        data :  Dictionary with keys equal to line numbers (starting at 0)
                Each line is itself a dictionary with the following entries:
                Key is line number
                "s1": Numpy float array with data of sensor 1
                "s2": Numpy float array with data of sensor 2
                      If only data of one single sensor were measured,
                      "s2" contains only one zero.
                "x":  Numpy float array with E-W coordinates of data points
                "y":  Numpy float array with N-S coordinates of data points
                "grad_flag" bool. True if 2 sensors were used, False if only one sensor
                "mask": bool, True if line should be plotted, False if excluded from
                              plotting. Initially set to True

        The original data are stored in class geometrics.Geometrics. See file
        geometrics.py for documentation

        """
        labels = [
            "Height of sensor 1 above ground [m]",
            ["Vertical disposition", "Horizontal disposition"],
            "Distance between sensors [m]\n"
            + "  positive: sensor 2 above or right of sensor 1",
            "Direction of lines [degrees from N to E]",
        ]
        types = ["e", "r", "e", "e"]
        values = [self.h_sensor, 0, self.d_sensor, self.line_declination]
        results, ok_button = dialog(
            labels, types, values, title="Geometrics parameters"
        )
        if not ok_button:
            print("Program aborted")
            sys.exit()
        self.h_sensor = float(results[0])
        self.dispo = int(results[1])
        self.d_sensor = float(results[2])
        self.line_declination = float(results[3])
        self.n_blocks += 1
        self.gdata.append(Geometrics())
        self.gdata[-1].read_stn(
            file, self.n_blocks, self.d_sensor, self.h_sensor, self.dispo
        )
        self.segments = self.gdata[-1].segments
        self.sensor1 = self.gdata[-1].sensor1
        self.sensor2 = self.gdata[-1].sensor2
        if len(self.sensor1) == len(self.sensor2):
            self.grad = (self.sensor2 - self.sensor1) / self.d_sensor
            self.grad_data = True
        else:
            self.grad = np.array([0.0])
            self.grad_data = False
        self.x = self.gdata[-1].x
        self.y = self.gdata[-1].y
        self.z = np.ones_like(self.x) * self.h_sensor
        self.topo = np.zeros_like(self.x)
        self.n_data = self.gdata[-1].n_data
        self.n_lines = self.gdata[-1].n_lines
        self.sensor1_ori = np.copy(self.sensor1)
        self.sensor2_ori = np.copy(self.sensor2)
        self.grad_ori = np.copy(self.grad)
        self.time = self.gdata[-1].time
        self.direction = 0
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        self.data = self.lines()
        self.data["grad_data"] = self.grad_data
        self.data["year"] = self.gdata[-1].year[0]
        self.data["dispo"] = self.dispo
        self.data["block"] = self.n_blocks
        self.data["height"] = self.h_sensor
        self.data["line_declination"] = self.line_declination
        if self.grad_data:
            self.data["height2"] = self.h_sensor + self.d_sensor
            self.data["d_sensor"] = self.d_sensor
        return self.data

    def write_geometrics(self, file, data1, x, y, data2=None, n_block=0):
        """
        Wrapper to write data in Geometrics MagMap2000 *.stn format
        Data must be interpolated onto a regular grid

        Parameters
        ----------
        file : str
            File name where to write data.
        data1 : numpy float array [number_of_samples_per_line, number_of_lines]
            Data of sensor 1.
        x : numpy float array [number_of_samples_per_line]
            X coordinates of all measured points.
        y : numpy float array [number_of_samples_per_line, number_of_lines]
            Y coordinates of all measured points.

        Optional parameters:
        data2 : numpy float array [number_of_samples_per_line, number_of_lines]
            Data of sensor 2. Optional. Default: np.zeros_like(data1)
        n_block : int, optional
            Number of block (data set) to be written. The default is 0.

        Returns
        -------
        None.

        """
        if not isinstance(data1, np.ndarray):
            n_block = data1["block"] - 1
        self.gdata[n_block].write_stn(file, data1, x, y, data2=data2)

    def read_txt(self, file):
        """
        Reads a non-Geometrics format magnetic data file
        This option is mainly thought for reading the output of program mgwin
        used with the option to enter all data points with their specific positions
        (NBPTS > 0). This allows calculation of a 2D map with mgwin. You may use
        Prepare_mgwin_calculation_points.py to preapre the coordinates.
        The file structure is as follows:
            One comment line
            line with NX, NY (number of points in X and Y direction)
                It is supposed that the data have been calculated on a regular
                grid. mgwin writes on this line the total number of data points
                the file must therefore be edited to replace the existing number
                by the two required ones.
            one line per data poin with (X, Y, Z, DATA)
            mgwin writes only one value into the file and in order to keep the
            structure of Geometrics simple, these values are copied into both
            sensor1 and sensor2 arrays.  Data are copied as well into
            self.sensor_n (1D array) and self.sensor_n_inter (2D array)

        Parameters
        ----------
        file : str, name of file to be read

        Returns
        -------
        data

        """
        labels = [
            "Height of lower sensor above ground [m]",
            "Distance between sensors [m]",
            "Direction of lines [degrees from N to E]",
        ]
        types = ["e", "e", "e"]
        values = [self.h_sensor, self.d_sensor, self.line_declination]
        results, ok_button = dialog(labels, types, values, title="MGWin parameters")
        if not ok_button:
            print("Program aborted")
            sys.exit()
        self.h_sensor = float(results[0])
        self.d_sensor = float(results[1])
        self.line_declination = float(results[2])
        self.n_blocks += 1
        with open(file, "r") as fi:
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
        # elif len(ll) <4:
        #     grad_flag = False
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
                # else:
                #     self.sensor2_inter[j,i] = self.sensor1_inter[j,i]
                #     self.sensor2.append(float(nums[3]))
            self.segments[line]["mark_samples"].append(n - 1)
            n2 = n - 2
            self.segments[line]["dx"].append(abs(self.x[-1] - self.x[-2]))
            self.segments[line]["dy"].append(abs(self.y[-1] - self.y[-2]))
            self.segments[line]["d"].append(
                np.sqrt(
                    self.segments[line]["dx"][-1] ** 2
                    + self.segments[line]["dy"][-1] ** 2
                )
            )
            self.segments[line]["median1"] = np.median(self.sensor1[n1:n2])
            self.segments[line]["median2"] = np.median(self.sensor2_inter[n1:n2])
            self.segments[line]["x"] = np.median(self.x[n1:n2])
            self.segments[line]["y"] = np.median(self.y[n1:n2])
            self.segments[line]["mask"] = True
            self.segments[line]["block"] = self.n_blocks
            self.segments[line]["dir"] = "odd"
            self.segments[line]["pos"] = self.segments[line]["x"]
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
        self.z = np.ones_like(self.x) * self.h_sensor
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
        self.data["block"] = self.n_blocks
        self.data["height"] = self.h_sensor
        self.data["line_declination"] = self.line_declination
        if grad_flag:
            self.data["height2"] = self.h_sensor + self.d_sensor
        return self.data

    def read_gxf(self, infile):
        """
        Read a gxf file (BRGM magnetic and gravity gridded files)
        input:
            infile: string
                    Name of file to be read
        """
        labels = [
            "Height of sensor above ground [m]",
            "Direction of lines [degrees from N to E]",
        ]
        types = ["e", "e"]
        values = [85.0, 0.0]
        results, ok_button = dialog(labels, types, values, title="MGWin parameters")
        if not ok_button:
            print("Program aborted")
            sys.exit()
        self.h_sensor = float(results[0])
        self.line_declination = float(results[1])
        with open(infile, "r") as fi:
            lines = fi.readlines()
        # Read header
        il = -1
        self.n_blocks += 1
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
                l = np.array(lines[il].split(), dtype=float)
                ic1 = ic2
                ic2 += len(l)
                data[ir, ic1:ic2] = l
                if ic2 == n_cols:
                    break
            data[data == dummy] = np.nan
            d = data[ir, :]
            index = np.isfinite(d)
            d = d[index]
            c2 = c1 + len(d)
            self.sensor1.extend(list(d))
            self.x.extend(list(self.x_inter[index]))
            self.y.extend(list(np.ones_like(self.x_inter[index]) * self.y_inter[ir]))
            self.segments[ir]["mark_samples"] = [c1, c2]
            self.segments[ir]["dx"] = [self.dx]
            self.segments[ir]["dy"] = [0]
            self.segments[ir]["d"] = [self.dx]
            self.segments[ir]["median1"] = np.median(d)
            self.segments[ir]["x"] = np.median(self.x_inter)
            self.segments[ir]["y"] = self.y_inter[ir]
            self.segments[ir]["mask"] = True
            self.segments[ir]["block"] = self.n_blocks
            self.segments[ir]["dir"] = "odd"
            self.segments[ir]["pos"] = self.y_inter[ir]
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.ones_like(self.x) * self.h_sensor
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
        self.data["block"] = self.n_blocks
        self.data["height"] = self.h_sensor
        self.data["line_declination"] = 0.0
        return self.data

    def read_BRGM_flight(self, file):
        """
        Reads magnetic data from flight lines out of a BRGM data file

        Parameters
        ----------
        file : str
            Name of file containing the data.
        line : int or float
            If > 100000. it is supposed that line contains the Easting coordinate in meters
                         rounded to the next km. The program transforms this coordinate
                         into the number of the flight line for Morvan data. First line at 666000.
            else: flight line number (starts with 320, increment 10, one line per km)

        Returns
        -------
        x :     1D numpy float array
                E coordinate of each measured point along the line [Lambert 2 extended, meters]
        y :     1D numpy float array
                N coordinate of each measured point along the line [Lambert 2 extended, meters]
        v :     1D numpy float array
                Magnetic anomaly (measured field minus IGRF) [nT]
        topo :  1D numpy float array
                DMT topography at each measured point along the line [m]
        height : 1D numpy float array
                Flight height above topo at each measured point along the line [m]
        num : int
                Number of flight line (same as line if line < 100000.)


        """
        self.n_blocks += 1
        line_number = []
        nl = -1
        x = []
        y = []
        t = []
        height = []
        v = []
        c2 = 0
        with open(file, "r") as fi:
            lines = fi.readlines()
        for _, l in enumerate(lines):
            if "/" in l:
                continue
            if "Line" in l:
                line_number.append(int(l.split()[1]))
                nl1 = nl
                nl += 1
                if nl == 0:
                    continue
                self.sensor1.extend(v)
                self.x.extend(x)
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
                self.segments[nl1]["median1"] = np.median(v)
                self.segments[nl1]["x"] = np.round(np.median(x), -3)
                self.segments[nl1]["y"] = np.median(y)
                self.segments[nl1]["mask"] = True
                self.segments[nl1]["block"] = self.n_blocks
                self.segments[nl1]["dir"] = "odd"
                self.segments[nl1]["pos"] = self.segments[nl1]["x"]
                x = []
                y = []
                t = []
                height = []
                v = []
            else:
                val = l.split()
                x.append(np.round(float(val[0]), -3))
                y.append(float(val[1]))
                v.append(float(val[14]))
                t.append(float(val[6]))
                height.append(float(val[4]))
        self.sensor1.extend(v)
        self.x.extend(x)
        self.y.extend(y)
        self.z.extend(height)
        self.topo.extend(t)
        self.segments[nl] = {}
        c1 = c2
        c2 += len(x)
        self.segments[nl]["mark_samples"] = [c1, c2]
        self.segments[nl]["dx"] = [1000.0]
        self.segments[nl]["dy"] = [abs(y[-1] - y[0]) / (len(y) - 1)]
        self.segments[nl]["d"] = [self.segments[nl]["dx"]]
        self.segments[nl]["median1"] = np.median(v)
        self.segments[nl]["x"] = np.round(np.median(x), 3)
        self.segments[nl]["y"] = np.round(np.median(y), 3)
        self.segments[nl]["mask"] = True
        self.segments[nl]["block"] = self.n_blocks
        self.segments[nl]["dir"] = "odd"
        self.segments[nl]["pos"] = self.segments[nl]["x"]
        self.n_lines = nl
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.z = np.array(self.z)
        self.topo = np.zeros_like(self.x)
        self.sensor1 = np.array(self.sensor1)
        # Store original data to arrays xxx_ori
        self.sensor1_ori = np.copy(self.sensor1)
        self.time = np.arange(len(self.x))
        self.data = self.lines()
        self.data["grad_data"] = False
        self.data["year"] = 0
        self.data["dispo"] = 0
        self.data["block"] = self.n_blocks
        self.data["height"] = np.mean(self.z)
        self.data["line_declination"] = 0.0
        return self.data

    def get_line(self, i_line=0):
        """
        Get data of one single line

        Parameters
        ----------
        i_line : int, optional
            Number of line to be extracted (counting starts at 0). The default is 0.

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
            may be "N", "S", "W" or "E", indicates the approximate line direction
        sensor: int
            if 0, two sensors are used in vertical configuration. 1 if only one
            sensor was used or sensor 1 in horizontal configuration. 2: sensor 2
            in horizontal configuration.
        median1: float
            Median value of line for data of sensor1.
        median2:0 float
            Median value of line for data of sensor2.

        """
        n1 = self.segments[i_line]["mark_samples"][0]
        n2 = self.segments[i_line]["mark_samples"][-1]
        if self.grad_data:
            return (
                self.sensor1[n1:n2],
                self.sensor2[n1:n2],
                self.x[n1:n2],
                self.y[n1:n2],
                self.z[n1:n2],
                self.time[n1:n2],
                self.segments[i_line]["mask"],
                self.segments[i_line]["direction"],
                self.segments[i_line]["sensor"],
                self.segments[i_line]["median1"],
                self.segments[i_line]["median2"],
            )
        dum = np.array([0])
        return (
            self.sensor1[n1:n2],
            dum,
            self.x[n1:n2],
            self.y[n1:n2],
            self.z[n1:n2],
            self.time[n1:n2],
            self.segments[i_line]["mask"],
            self.segments[i_line]["direction"],
            self.segments[i_line]["sensor"],
            self.segments[i_line]["median1"],
            self.segments[i_line]["median2"],
        )

    def lines(self):
        """
        Put all data into a simplified dictionary, one entry per line

        No input parameters

        Returns
        -------
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
        """
        data = {}
        for i, _ in self.segments.items():
            s1, s2, x, y, z, t, m, d, s, med1, med2 = self.get_line(i)
            data[i] = {}
            data[i]["s1"] = s1
            data[i]["s2"] = s2
            data[i]["x"] = x
            data[i]["y"] = y
            data[i]["z"] = z
            data[i]["time"] = t
            data[i]["mask"] = m
            data[i]["direction"] = d
            data[i]["sensor"] = s
            data[i]["median1"] = med1
            data[i]["median2"] = med2
        return data

    def write_dat(self, file, data1, x, y, *, data2=None, z=None):
        """
        Writes Geometrics magnetic gradiometer file in Surfer format.

        Parameters
        ----------
        Obligatory parameters
        file : str
            File where to store the data.
        data1 : numpy 1D float array
                Contains all data of sensor 1
        x :     numpy 1D float array
                Contains E-W coordinates of all data points
        y :     numpy 1D float array
                Contains N-S coordinates of all data points

        Optional parameters
        data2 : numpy 1D float array
                Contains all data of sensor 2 if there are. Default: None
                (must have the same shape as data1)
        z :     numpy 1D float array
                Topography of all data points
                If it is None, z-coordinate is set to zero

        Returns
        -------
        None.

        """
        with open(file, "w") as fo:
            fo.write(
                "           X           Y           Z     TOP_RDG  "
                + "BOTTOM_RDG    VRT_GRAD\n"
            )
            # Write gridded data
            if isinstance(data1, np.ndarray):
                if not isinstance(z, np.ndarray):
                    z = np.zeros_like(data1)
                ny, nx = data1.shape
                if not isinstance(data2, np.ndarray):
                    for ix in range(nx):
                        xx = x[ix]
                        for iy in range(ny):
                            fo.write(
                                f"{xx:12.3f}{y[iy]:12.3f}"
                                + f"{z[iy,ix]:12.3f}{data1[iy,ix]:12.3f}\n"
                            )
                else:
                    for ix in range(nx):
                        xx = x[ix]
                        for iy in range(ny):
                            fo.write(
                                f"{xx:12.3f}{y[iy]:12.3f}"
                                + f"{z[iy,ix]:12.3f}{data1[iy,ix]:12.3f}"
                                + f"{data2[iy,ix]:12.3f}"
                                + f"{data2[iy,ix]-data1[iy,ix]:12.3f}\n"
                            )
            # Write non-gridded data
            else:
                for key, val in data1.items():
                    if isinstance(key, str):
                        break
                    if data1["grad_data"]:
                        for i, xx in enumerate(val["x"]):
                            fo.write(
                                f"{xx:12.3f}{val['y'][i]:12.3f}"
                                + f"{val['z'][i]:12.3f}{val['s1'][i]:12.3f}"
                                + f"{val['s2'][i]:12.3f}{val['s2'][i]-val['s1'][i]:12.3f}\n"
                            )
                    else:
                        for i, xx in enumerate(val["x"]):
                            fo.write(
                                f"{xx:12.3f}{val['y'][i]:12.3f}"
                                + f"{val['z'][i]:12.3f}{val['s1'][i]:12.3f}\n"
                            )
        print(f"\nfile {file} written")

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
        self.base = Geometrics()
        print(f"\nRead base station file {file}")
        self.base.read_base(file, year=year)

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
