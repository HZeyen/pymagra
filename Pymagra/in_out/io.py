# -*- coding: utf-8 -*-
"""
Last modified on Apr 18, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         University Paris-Saclay, France

Contains functions for data input/output

Contains methods:
    - get_files
    - read_lineaments
    - read_synthetic_model
    - read_geography_file
    - get_mag_field
    - read_geometrics
    - write_geometrics

"""

#        fill_nans (actually not used)
#            eliminate_nans (actually not used)
import sys
import os
import numpy as np
from PyQt5 import QtWidgets

# from ..plotting import plot as w
from .dialog import dialog
from ..data.geometrics import Geometrics
from .earth import Earth_mag


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
    ftypes = ("GEOMETRICS", "MGWIN", "BRGM", "GXF", "GEOMETRICS")
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
    data_types = []
    file_conf = []
    for _, f in enumerate(files[0]):
        _, file_ext = os.path.splitext(f)
        file_ext = file_ext.upper()
        if os.path.isdir(f):
            continue
        if file_ext not in valid_extensions:
            continue
        data_files.append(f)
        fconfig = os.path.basename(f)
        j = fconfig.rfind(".")
        # Check whether there is a configuration file for each data file
        if j > 0:
            fconfig = fconfig[:j] + ".config"
        else:
            fconfig += ".config"
        # If there is, read its content
        if os.path.isfile(fconfig):
            with open(fconfig, "r") as fc:
                file_types.append(fc.readline()[:-1].upper())
                data_types.append(fc.readline()[:-1])
            file_conf.append(True)
        else:
            file_conf.append(False)
            data_types.append("")
            n_ftype = np.where(valid_extensions == file_ext)[0][0]
            if n_ftype == len(ftypes) - 1:
                n_ftype = 0
            file_types.append(ftypes[n_ftype])
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
        for _ in range(len(files[0])):
            data_types.append("magnetic")
    else:
        for i, f in enumerate(data_files):
            if file_conf[i]:
                continue
            labels.append(f"{os.path.basename(f)}:")
            values.append(None)
            types.append("l")
            labels.append(["Magnetic", "Gravity"])
            if "gra" in f.lower() or "bou" in f.lower():
                values.append("2")
            else:
                values.append("1")
            types.append("r")
            labels.append("\nfile type:")
            values.append(None)
            types.append("l")
            labels.append(ftypes[:-1])
            types.append("r")
            values.append(n_ftype + 1)
        if len(labels) > 0:
            results, ok_button = dialog(labels, types, values, title="data types")
            if not ok_button:
                print("No entry, program finished")
                sys.exit()
            ir = -1
            for i, flag in enumerate(file_conf):
                if flag:
                    continue
                ir += 2
                data_types[i] = dtypes[int(results[ir])]
                ir += 2
                file_types[i] = ftypes[int(results[ir])]
    return data_files, file_types, data_types, dir0


def read_lineaments(file="lineaments.dat"):
    """
    Reads file with picked lineament information (usually done on tilt angle
    plots)

    Parameters
    ----------
    file : str, optional. Default: "lineaments.dat"
        Name of file to be read

    Returns
    -------
    lineaments : dictionary
        The following entries are available:

        - "x" : numpy 1D float array
          Contains x coordinates of lineaments (E-W direction)
        - "y" : numpy 1D float array
          Contains y coordinates of lineaments (N-S direction)
        - "type" : str
          Type of lineament. May be "magnetic" or "gravity"

    """
    with open(file, "r", encoding="utf-8") as fi:
        lines = fi.readlines()
    il = 0
    nlineaments = 0
    x = []
    y = []
    lineaments = {}
    while True:
        if lines[il][0] == "#":
            if len(x) > 0:
                lineaments[nlineaments]["x"] = np.array(x)
                lineaments[nlineaments]["y"] = np.array(y)
                x = []
                y = []
            if lines[il][:4] == "#END":
                break
            nlineaments += 1
            lineaments[nlineaments] = {}
            lineaments[nlineaments]["type"] = lines[il][1:-1]
            il += 1
        else:
            nums = lines[il].split()
            x.append(float(nums[0]))
            y.append(float(nums[1]))
            il += 1
    return lineaments


def read_synthetic_model():
    """
    Read synthetic model.
    The file should have an extension .txt, .dat or .mod
    The model is composed of rectangular prisms with faces parallel to axis.
    The format of the file is as follows:

    - No header line
    - One line per prism to be calculated containing 7 to 11 values each:
      xmin, xmax, ymin, ymax, zmin, zmax, sus, rem_s, rem_i, rem_d, rho

    - xmin, xmax: minimum and maximum x_coordinates (E-W) of prism [m]
    - ymin, ymax: minimum and maximum y_coordinates (N-S) of prism [m]
    - zmin, zmax: minimum and maximum z_coordinates (positive down) of
      prism [m]
    - sus: susceptibility [SI units]
    - rem_s: intensity of remanent magnetization [A/m]
    - rem_i: inclination of remanent magnetization [degrees]
    - rem_d: declination of remanent magnetization [degrees]
    - rho: density of prism [kg/m3]

    Returns
    -------
    x : numpy float array of shape (n_prisms, 2)
        X-coordinates of prisms.
    y : numpy float array of shape (n_prisms, 2)
        Y-coordinates of prisms.
    z : numpy float array of shape (n_prisms, 2)
        Z-coordinates of prisms.
    sus : numpy float array of shape (n_prisms)
        Susceptibilities of prisms.
    rem : numpy float array of shape (n_prisms)
        Remanence intensities of prisms.
    rem_i : numpy float array of shape (n_prisms)
        Remanence inclinations of prisms.
    rem : numpy float array of shape (n_prisms)
        Remanence declinations of prisms.
    rho : numpy float array of shape (n_prisms)
        Densities of prisms.

    """
    file = list(
        QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Select model file",
            "",
            filter="txt/dat/mod (*.txt *.dat *.mod) ;; all (*.*)",
        )
    )
    if len(file) == 0:
        print("No file chosen, program finishes")
        return None, None, None, None, None, None, None, None
    if len(file[0]) < 1:
        print("read_synthetic_model: No files read")
        return None, None, None, None, None, None, None, None
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    zmin = []
    zmax = []
    sus = []
    rem = []
    rem_i = []
    rem_d = []
    rho = []
    with open(file[0], "r") as fi:
        lines = fi.readlines()
    for line in lines:
        val = line.split()
        ncol = len(val)
        if ncol < 7:
            answer = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                "Synthetic model file does not have enough columns:\n"
                + f"At least 7 columns are needed, {ncol} found.\n"
                + "Synthetic modeling aborted.",
                QtWidgets.QMessageBox.Close,
                QtWidgets.QMessageBox.Ignore,
            )
            return None, None, None, None, None, None, None, None
        if ncol < 11:
            if ncol == 7:
                text = "Remanence and density are set to zero."
            else:
                text = "Density is set to zero."
            answer = QtWidgets.QMessageBox.warning(
                None,
                "Warning",
                f"Synthetic model file has only {ncol} columns:\n"
                + f"{text}\nPress Ignore to accept or Abort to abandon.",
                QtWidgets.QMessageBox.Ignore | QtWidgets.QMessageBox.Abort,
                QtWidgets.QMessageBox.Ignore,
            )
            if answer == QtWidgets.QMessageBox.Abort:
                return None, None, None, None, None, None, None, None
        xmin.append(float(val[0]))
        xmax.append(float(val[1]))
        ymin.append(float(val[2]))
        ymax.append(float(val[3]))
        zmin.append(float(val[4]))
        zmax.append(float(val[5]))
        sus.append(float(val[6]))
        if ncol > 7:
            rem.append(float(val[7]))
            if ncol > 8:
                rem_i.append(float(val[8]))
                if ncol > 9:
                    rem_d.append(float(val[9]))
                    if ncol > 10:
                        rho.append(float(val[10]))
                    else:
                        rho.append(0.0)
                else:
                    rem_d.append(0.0)
                    rho.append(0.0)
            else:
                rem_i.append(0.0)
                rem_d.append(0.0)
                rho.append(0.0)
        else:
            rem.append(0.0)
            rem_i.append(0.0)
            rem_d.append(0.0)
            rho.append(0.0)
        nprism = len(xmin)
        x = np.zeros((nprism, 2))
        x[:, 0] = np.array(xmin)
        x[:, 1] = np.array(xmax)
        y = np.zeros((nprism, 2))
        y[:, 0] = np.array(ymin)
        y[:, 1] = np.array(ymax)
        z = np.zeros((nprism, 2))
        z[:, 0] = np.array(zmin)
        z[:, 1] = np.array(zmax)
    return (
        x,
        y,
        z,
        np.array(sus),
        np.array(rem),
        np.array(rem_i),
        np.array(rem_d),
        np.array(rho),
    )


def read_geography_file(file):
    """
    Reads file with geography information to be plotted
    (borders - mainly geological, towns)

    Parameters
    ----------
    file : str, name of file to be read
        File has the following structure:

        - keyword may be "#POINT", "#LINE" or "#END"
        - if keyword == "#POINT", one line follows with x, y coordinates and
          text, text being the description of the point (no blanks)
        - if keyword == "#LINE", several lines follow, each one with x and y
          coordinate of one point describing the line
        - if keyword == "#END", this line finishes the data entry, possible
          following lines will be ignored.

    Returns
    -------
    geography : dictionary with all geography information.
        key is consecutive numbering of entries
        Each entry consists of a dictionary with the following entries:

        - "type" str
          may be "POINT" or "LINE"

          - If type == "POINT" : One line with:

              - "x" : float: x coordinate of point (East)
              - "y" : float: y coordinate of point (North)
              - "name" : str: Text to be plotted beside the point mostly name
                of a town

          - If type == "line" :

              - "x" : list of floats, East coordinate of points describing
                the line
              - "y" : list of floats, North coordinate of points describing
                the line

    """
    with open(file, "r", encoding="utf-8") as fi:
        ll = fi.readlines()
    geography = {}
    il = 0
    iunit = -1
    while True:
        if ll[il].upper().startswith("#POINT"):
            iunit += 1
            il += 1
            nums = ll[il].split()
            geography[iunit] = {}
            geography[iunit]["type"] = "POINT"
            geography[iunit]["x"] = float(nums[0])
            geography[iunit]["y"] = float(nums[1])
            geography[iunit]["name"] = nums[2]
            il += 1
        elif ll[il].upper().startswith("#LINE"):
            iunit += 1
            geography[iunit] = {}
            geography[iunit]["type"] = "LINE"
            geography[iunit]["x"] = []
            geography[iunit]["y"] = []
            while True:
                il += 1
                if ll[il].startswith("#"):
                    break
                nums = ll[il].split()
                geography[iunit]["x"].append(float(nums[0]))
                geography[iunit]["y"].append(float(nums[1]))
        elif ll[il].upper().startswith("#END"):
            break
    return geography


def get_mag_field(line_dir, strength=None, inclination=None, declination=None):
    """
    Get parameters of Earth's magnetic field in the study area
    The declination is calculated as absolute declitation minus line direction

    Parameters
    ----------
    line_dir : float
        Direction of Y axis with respect to North (positive towards E, degree)
    strength: float, optional. Default: NOne
        Field strength [nT]. If None, the field parameters are asked for
        interactively
    inclination: float, optional. Default: NOne
        Field inclination [degrees].
    declination: float, optional. Default: NOne
        Field declination [degrees] with respect to geographic north.

    Returns
    -------

    inclination : float
        Inclination of Earth's field in degrees
    declination : float
        Declination of Earth's field in degrees
    strength : float
        Strength of magnetic field [nT]
    """
    if strength is None:
        results, _ = dialog(
            ["Field strength [nT]", "Field inclination", "Field declination"],
            ["e", "e", "e"],
            [50000, 62, 0],
            "Earth's field parameters",
        )
        strength = float(results[0])
        inclination = float(results[1])
        declination = float(results[2]) - line_dir
    else:
        declination -= line_dir
    earth = Earth_mag(strength, inclination, declination)
    return earth


def read_geometrics(file, n_block, height1, height2, dispo):
    """
    Read Geometrics .stn or .dat file (G-858 instrument)

    Parameters
    ----------
    file : str
        Name of data file.
    n_block : int
        Number of block to be read
    height1 : float
        Height of sensor 1 above ground (meters)
    height2 : float
        Height of sensor 2 above ground (meters)
    dispo : int
        Disposition of sensors if there are two sensors:
        0: vertical disposition, 1: horizontal disposition

    Returns
    -------
    gdata :  Dictionary with keys equal to line numbers (starting at 0)
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
    gdata = Geometrics()
    if ".stn" in file:
        gdata.read_stn(file, n_block, height1, height2, dispo)
    else:
        gdata.read_dat(file, n_block, height1, height2, dispo)
        os.remove("temp.stn")
    return gdata


def write_geometrics(self, file, data1, x, y, data2=None, n_block=0, time=None):
    """
    Wrapper to write data in Geometrics MagMap2000 .stn format.

    Data must be interpolated onto a regular grid.

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
    n_block = 0
    self.gdata[n_block].write_stn(file, data1, x, y, data2=data2, time=time)
