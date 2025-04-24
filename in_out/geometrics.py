# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 2023, 15:12:59
Last modified on Mon Oct 30 2023

@author: Hermann Zeyen University Paris-Saclay, France
"""

import numpy as np
from PyQt5 import QtWidgets
from ..utilities import Utilities


class Geometrics:
    """
    Methods:
        __init__
        fill_dict
        read_stn
        write_stn
        clean_data
        line_days
        get_line
        geometrics_lines
        get_segment
        read_base
        write_base

    """

    def __init__(self):
        """
        Class Geometrics reads and writes Geometrics .stn files and allows
        a number of data treatments

        Parameters
        ----------

        Returns
        -------
        None.

        """
        # Initialize main frame widget
        #        super(Geometrics, self).__init__()
        super().__init__()
        self.u = Utilities(self)
        self.n_lines = 0
        self.n_data = 0
        self.grad_data = True
        self.geography_flag = False
        self.grid_flag = True
        self.unit = "nT"
        # \        self.read_stn(file)
        self.lines_per_day = {}
        self.grad = np.array([])
        self.year_base = np.array([])
        self.month_base = np.array([])
        self.day_base = np.array([])
        self.hour_base = np.array([])
        self.minute_base = np.array([])
        self.second_base = np.array([])
        self.time_base = np.array([])
        self.jday_base = np.array([])
        self.base = np.array([])
        self.year = np.array([])
        self.month = np.array([])
        self.day = np.array([])
        self.hour = np.array([])
        self.minute = np.array([])
        self.second = np.array([])
        self.time = np.array([])
        self.x = np.array([])
        self.y = np.array([])
        self.x1 = np.array([])
        self.y1 = np.array([])
        self.x2 = np.array([])
        self.y2 = np.array([])
        self.sensor1 = np.array([])
        self.sensor2 = np.array([])
        self.sensor1_ori = np.array([])
        self.sensor2_ori = np.array([])
        self.grad_ori = np.array([])
        self.n_blocks = 0
        self.d_sensor = 0.0
        self.d_sensor2 = 0.0
        self.h_sensor = 0.0
        self.dispo = 0
        self.infile = ""
        self.segments = {}
        self.grad_flag = False
        self.dx = 0.0
        self.dy = 0.0
        self.direction = 0
        self.d_lines = 0.0
        self.line_pos = np.array([])
        self.line_choice = "all"

    def fill_dict(
        self, ntracks, x1, y1, mag1, x2, y2, mag2, mark_samples, dx_seg, dy_seg, d_seg
    ):
        """
        Fill in values of dictionary self.segments for a full line.

        Parameters
        ----------
        ntracks : int
            Actual line number (in horizontal configuration, dictionary entrancies
                                ntracks and ntarcks+1 are filled).
        x1 : list of floats
            X-coordinates for sensor 1 data.
        y1 : list of floats
            Y-coordinates for sensor 1 data.
        mag1 : list of floats
            Magnetic data recorded by sensor 1.
        x2 : list of floats
            X-coordinates for sensor 2 data.
        y2 : list of floats
            Y-coordinates for sensor 2 data.
        mag2 : list of floats
            Magnetic data recorded by sensor 1.
        mark_samples : list of int
            Number of the sample just after a mark.
        dx_seg : list of floats
            Step length in x-direction for every segment.
        dy_seg : list of floats
            Step length in y-direction for every segment.
        d_seg : list of floats
            Total step length for every segment.

        Returns
        -------
        None.

        """
        dx = x1[-1] - x1[0]
        dy = y1[-1] - y1[0]
        nt1 = ntracks + 1
        if ntracks == 0:
            nshift = 0
        else:
            nshift = self.segments[ntracks - 1]["mark_samples"][-1]
        if abs(dx) > abs(dy):
            if dx > 0:
                self.segments[ntracks]["direction"] = "E"
            else:
                self.segments[ntracks]["direction"] = "W"
        else:
            if dy > 0:
                self.segments[ntracks]["direction"] = "N"
            else:
                self.segments[ntracks]["direction"] = "S"
        self.segments[ntracks]["median1"] = np.median(mag1)
        self.segments[ntracks]["mask"] = True
        self.segments[ntracks]["block"] = self.n_blocks
        self.segments[ntracks]["mark_samples"] = (
            np.array(mark_samples, dtype=int) + nshift
        )
        self.segments[ntracks]["dx"] = np.array(dx_seg)
        self.segments[ntracks]["dy"] = np.array(dy_seg)
        self.segments[ntracks]["d"] = np.array(d_seg)
        self.segments[ntracks]["x"] = np.median(x1)
        self.segments[ntracks]["y"] = np.median(y1)
        if self.dispo == 0:
            self.segments[ntracks]["median2"] = np.median(mag2)
            self.segments[ntracks]["sensor"] = 0
        else:
            self.segments[ntracks]["sensor"] = 1
            self.segments[ntracks]["median2"] = 0.0
            self.segments[nt1]["median1"] = np.median(mag2)
            self.segments[nt1]["median2"] = 0.0
            self.segments[nt1]["x"] = np.median(x2)
            self.segments[nt1]["y"] = np.median(y2)
            self.segments[nt1]["mask"] = True
            self.segments[nt1]["block"] = self.n_blocks
            self.segments[nt1]["sensor"] = 2
            self.segments[nt1]["direction"] = self.segments[ntracks]["direction"]
            self.segments[nt1]["mark_samples"] = (
                np.array(mark_samples, dtype=int)
                + self.segments[ntracks]["mark_samples"][-1]
            )
            self.segments[nt1]["dx"] = np.array(dx_seg)
            self.segments[nt1]["dy"] = np.array(dy_seg)
            self.segments[nt1]["d"] = np.array(d_seg)

    def append_data(self, x, y, d, year, month, day, hour, minute, second, time):
        """
        Appends date and time information of one segment to global arrays

        Parameters
        ----------
        x: list of float
            x-coordinate of data points
        y: list of float
            y-coordinate of data points
        d: list of float
            measured values of sensor 1
        year: list of int
            year of acquisition of data points
        month: list of int
            month of acquisition of data points
        day: list of int
            day of acquisition of data points
        hour: list of int
            hour of acquisition of data points
        minute: list of int
            minute of acquisition of data points
        second: list of float
            second of acquisition of data points
        time: list of int
            julian day of data points
        """
        self.x = np.concatenate((self.x, np.round(np.array(x, dtype=float), 3)))
        self.y = np.concatenate((self.y, np.round(np.array(y, dtype=float), 3)))
        self.sensor1 = np.concatenate((self.sensor1, np.array(d, dtype=float)))
        self.day = np.concatenate((self.day, np.array(day, dtype=int)))
        self.month = np.concatenate((self.month, np.array(month, dtype=int)))
        self.year = np.concatenate((self.year, np.array(year, dtype=int)))
        self.hour = np.concatenate((self.hour, np.array(hour, dtype=int)))
        self.minute = np.concatenate((self.minute, np.array(minute, dtype=int)))
        self.second = np.concatenate((self.second, np.array(second, dtype=float)))
        tt = (
            np.array(time, dtype=int) * 86400.0
            + np.array(hour, dtype=int) * 3600.0
            + np.array(minute, dtype=int) * 60.0
            + np.array(second, dtype=float)
        )
        self.time = np.concatenate((self.time, tt))

    def read_stn(self, infile, n_block, d_sensor=0.9, h_sensor=0.4, dispo=0):
        """
        Reads a 2-sensor Geometix .stn file
        If other data have already been read from another file, the new ones are
        added to the vectors with new line numbers, even if effectively, the
        measurements may have been done in the prolongation of existing lines.

        Parameters
        ----------
        infile : str
            name of file to be read
        n_block : int
            Number of block (file) of this data set within all files to be read
        d_sensor : float, optional
            distance between sensors [m]. It is supposed that sensor1 is the
            upper one. If this is not the case, d_sensor must be negative.
            Default: 0.9 m.
        h_sensor : float, optional
            Hight of lower sensor (sensor 2) above ground [m]. Default: 0.4 m
        disop : int, optional
            If two sensors were used, dispo indicates there relative position:
                If dispo == 0: vertical disposition, sensor 2 above sensor 1
                If dispo == 1: horizontal disposition, sensor 2 to the right of sensor 1


        Data are stored in the following arrays:
            self.sensor1 : 1D numpy float array contains all data of sensor 1
            self.sensor2 : 1D numpy float array contains all data of sensor 2
            self.day     : 1D numpy int array contains day of acquisition for all data
            self.month   : 1D numpy int array contains month of acquisition for all data
            self.year    : 1D numpy int array contains year of acquisition for all data
            self.hour    : 1D numpy int array contains hour of acquisition for all data
            self.minute  : 1D numpy int array contains minute of acquisition for all data
            self.second  : 1D numpy float array contains second of acquisition for all data
            self.time    : 1D numpy float array contains seconds of acquisition for all data
                                    counted from the beginning of the acquisition year
            self.x       : 1D numpy float array contains x-coordinates of measurement points
            self.y       : 1D numpy float array contains y-coordinates of measurement points
            self.segments: Dictionary with information for all lines and segments measured
                           Since all data are stored sequentially, we need to know
                           which data belong to which line and it may also be
                           necessary to know where marks were set, thus where a
                           measured segment starts and ends.
                           self.segments.keys() contains line numbers starting at zero
                           Every dictionary entrance contains the following lists:
                               "mark_samples" contains for every mark (starting and
                                              end of line included) the number of
                                              the sample in self.sensor1 or 2
                                              recorded just after the mark
                                "mask" is initially set to True but may be modified
                                       in order to plot only certain lines. If True,
                                       lines are plotted, else they are excluded from plot
                                "median1" : median value of sensor 1 for a line
                                "median2" : median value of sensor 2 for a line
                                            If sensors are in horizontal configuration,
                                            both sensors are treated as sensor 1 and
                                            "median2" has value 0.
                                "sensor" : may be 0 if both sensors are used in vertical
                                            configuration, 1 or 2 if line has been measured
                                            by sensor 1 or 2 in horizontal configuration
                                "dx" contains for every segment the sampling step
                                     in X direction
                                "dy" contains for every segment the sampling step
                                     in y direction
                                "d" contains for every segment the sampling step length
                                dx, dy and d contain one value less than mark_samples
                                "x" median of x-coordinates
                                "y" median of y-coordiantes
                                    depending on the line direction, "x" or "y" are
                                    supposed to be the position of the line
                                "pos" coordinate of the line
                                "block" number of block (or file) if several files
                                        are read
                                "dir" may be "odd" or "even". First measured line
                                      is "odd" (odd or even refers to natural counting)
                                "direction" is "N", "S", "W" or "E" depending on the
                                            line direction
                In addition, the following attributes are accessible:
                    self.n_data   : int, total number of data
                    self.n_lines : int, number of lines read
                    self.direction : int, direction of measurement lines.
                                    May be 0 (Y-direction) or 1 (X-direction)
                    self.d_lines : float, distance between measurement lines
        Returns
        -------
        data: dictionary with one entry per line, key = number of line.
              Each entry is itself a dictionary containing the following entries:
              "s1": Numpy float array with data of sensor 1
              "s2": Numpy float array with data of sensor 2
                    If only data of one single sensor were measured,
                    "s2" contains only one zero.
              "x":  E-W coordinates of data points
              "y"   N-S coordinates of data points

        """
        # Read magnetic field data ("stn" file)
        self.n_blocks = n_block
        self.d_sensor = d_sensor
        self.d_sensor2 = d_sensor / 2.0
        self.h_sensor = h_sensor
        self.dispo = dispo
        self.infile = infile
        mag1 = []
        mag2 = []
        xpt = []
        ypt = []
        xpt1 = []
        ypt1 = []
        xpt2 = []
        ypt2 = []
        year_m = []
        month_m = []
        day_m = []
        hour_m = []
        min_m = []
        sec_m = []
        time_m = []
        if self.n_lines == 0:
            self.sensor1 = np.array([])
            self.sensor2 = np.array([])
            self.x = np.array([])
            self.y = np.array([])
            self.x1 = np.array([])
            self.y1 = np.array([])
            self.x2 = np.array([])
            self.y2 = np.array([])
            self.day = np.array([], dtype=int)
            self.month = np.array([], dtype=int)
            self.year = np.array([], dtype=int)
            self.hour = np.array([], dtype=int)
            self.minute = np.array([], dtype=int)
            self.second = np.array([])
            self.time = np.array([])
            self.segments = {}
            n_start = 0
            first = True
        else:
            n_start = len(self.sensor1)
            first = False

        ntracks = self.n_lines - 1

        # read all lines in magnetic data file
        with open(infile, "r") as fh:
            ll = fh.readlines()
        try:
            _ = int(ll[0].split()[0])
        except Exception as e:
            # If none of the above, stop program
            _ = QtWidgets.QMessageBox.critical(
                None,
                "Error",
                f"Error: {e}\n\n"
                + f"File {infile} is not a Geometrics file.\nProgram stops",
                QtWidgets.QMessageBox.Ok,
            )
            raise IOError("Wrong file type\n") from e
        #            sys.exit()
        # store reversed order in list "lines"
        lines = ll[::-1]
        nlines = len(lines)
        last_mark = 5
        ndat_seg = 0
        xx0 = 0.0
        yy0 = 0.0
        ddx = 0.0
        ddy = 0.0
        nstart = True
        nt1 = ntracks
        mark_samples = []
        dx_seg = []
        dy_seg = []
        d_seg = []

        # read the lines and get data values and corresponding coordinates
        for k in range(nlines):
            l = lines[k].split()
            # if first number in line is 3, it is a mark (including beginning and end of line)
            #    seg_break_data contains number of data point before which mark is located
            #    id_seg_break contains the type of mark
            #    seg_break_line contains the line in the data file where mark is located
            #    line segments contains for every line the position of marks (between marks,
            #                  data positions are interpolated at regular spacing)
            #    dx_seg contains distance between samples for every segment
            if int(l[0]) == 3:
                xx = np.round(float(l[1]), 3)
                yy = np.round(float(l[2]), 3)
                # code 36 indicates beginning of a new line
                # code 4 indicates a mark, 5 is the Endline mark
                if int(l[8]) in (4, 5):
                    if ndat_seg == 0:
                        continue
                    mark_samples.append(len(mag1) + n_start)
                    n1 = mark_samples[-2]
                    n2 = mark_samples[-1]
                    ndat_segment = n2 - n1
                    if ndat_segment == 0:
                        del mark_samples[-1]
                        continue
                    # print(f"nline: {ntracks}: nseg: "+\
                    #       f"{len(self.segments[ntracks]['mark_samples'])-1},"+\
                    #       f" ndseg: {ndat_segment}")
                    # If last mark had code 36, a new line was started and it is the first mark after start,
                    #    The number of measurement steps is 1 less than the number of data points, if not,
                    #    both are equal. This means that it is supposed that the mark was pressed at
                    #    the position of the measurement just before the mark
                    if nstart:
                        dx = (xx - xx0) / (ndat_segment - 1)
                        dy = (yy - yy0) / (ndat_segment - 1)
                        n2 -= 1
                    else:
                        dx = (xx - xx0) / ndat_segment
                        dy = (yy - yy0) / ndat_segment
                    d = np.sqrt(dx * dx + dy * dy)
                    dx_seg.append(dx)
                    dy_seg.append(dy)
                    d_seg.append(d)
                    if self.dispo > 0:
                        ddx1 = ddx
                        ddy1 = ddy
                        nt1 = ntracks + 1
                        ddx = dy / d * self.d_sensor2
                        ddy = dx / d * self.d_sensor2
                    else:
                        nt1 = ntracks
                        ddx1 = 0.0
                        ddy1 = 0.0
                        ddx = 0.0
                        ddy = 0.0
                    if nstart:
                        xpt1[0] -= ddx
                        ypt1[0] += ddy
                        xpt2[0] += ddx
                        ypt2[0] -= ddy
                    else:
                        xpt1.append(xpt1[-1] + dx - ddx + ddx1)
                        ypt1.append(ypt1[-1] + dy + ddy - ddy1)
                        xpt2.append(xpt2[-1] + dx + ddx - ddx1)
                        ypt2.append(ypt2[-1] + dy - ddy + ddy1)
                        n1 += 1
                    for _ in range(n1, n2):
                        xpt1.append(xpt1[-1] + dx)
                        ypt1.append(ypt1[-1] + dy)
                        xpt2.append(xpt2[-1] + dx)
                        ypt2.append(ypt2[-1] + dy)
                    last_mark = int(l[8])
                    xx0 = xx
                    yy0 = yy
                    nstart = False
                if int(l[8]) == 5 or (int(l[8]) == 36 and last_mark != 5):
                    self.fill_dict(
                        ntracks,
                        xpt1,
                        ypt1,
                        mag1,
                        xpt2,
                        ypt2,
                        mag2,
                        mark_samples,
                        dx_seg,
                        dy_seg,
                        d_seg,
                    )
                    self.append_data(
                        xpt1,
                        ypt1,
                        mag1,
                        year_m,
                        month_m,
                        day_m,
                        hour_m,
                        min_m,
                        sec_m,
                        time_m,
                    )
                    if self.dispo:
                        self.append_data(
                            xpt2,
                            ypt2,
                            mag2,
                            year_m,
                            month_m,
                            day_m,
                            hour_m,
                            min_m,
                            sec_m,
                            time_m,
                        )
                        self.n_lines += 1
                    else:
                        self.sensor2 = np.concatenate(
                            (self.sensor2, np.array(mag2, dtype=float))
                        )
                    last_mark = int(l[8])
                    ndat_seg = 0
                # xpt and ypt contain the coordinates of all data points of the actual line
                if int(l[8]) == 36:
                    xx0 = xx
                    yy0 = yy
                    xpt1 = [xx0]
                    ypt1 = [yy0]
                    xpt2 = [xx0]
                    ypt2 = [yy0]
                    dx_seg = []
                    dy_seg = []
                    d_seg = []
                    mag1 = []
                    mag2 = []
                    time_m = []
                    month_m = []
                    day_m = []
                    hour_m = []
                    min_m = []
                    sec_m = []
                    ndat_segment = int(l[5])
                    ndat_seg = 0
                    nstart = True
                    ntracks = nt1 + 1
                    self.n_lines += 1
                    self.segments[ntracks] = {}
                    mark_samples = [0]
                    if self.dispo:
                        self.segments[ntracks + 1] = {}
                    continue
                # Don't take into account other mark codes (mainly Pause)
                continue
            # if first number in line is 0, it is a data point
            # mag1 contains data of first sensor, mag 2 of second sensor
            # In addition, times are stored
            if int(l[0]) == 0:
                mag1.append(float(l[1]))
                mag2.append(float(l[2]))
                t = l[3].split(":")
                hour_m.append(int(t[0]))
                min_m.append(int(t[1]))
                sec_m.append(float(t[2]))
                t = l[4].split("/")
                day_m.append(int(t[1]))
                month_m.append(int(t[0]))
                year_m.append(int(t[2]))
                time_m.append(self.u.date2julian(day_m[-1], month_m[-1], year_m[-1]))
                ndat_seg += 1

        if self.n_lines > 0 and last_mark != 5:
            self.fill_dict(
                ntracks,
                xpt1,
                ypt1,
                mag1,
                xpt2,
                ypt2,
                mag2,
                mark_samples,
                dx_seg,
                dy_seg,
                d_seg,
            )
            self.append_data(
                xpt1, ypt1, mag1, year_m, month_m, day_m, hour_m, min_m, sec_m, time_m
            )
            if self.dispo > 0:
                self.append_data(
                    xpt2,
                    ypt2,
                    mag2,
                    year_m,
                    month_m,
                    day_m,
                    hour_m,
                    min_m,
                    sec_m,
                    time_m,
                )
            else:
                self.sensor2 = np.concatenate(
                    (self.sensor2, np.array(mag2, dtype=float))
                )
        self.n_data += len(mag1)
        if len(self.sensor1) == len(self.sensor2):
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor
            self.grad_flag = True
        else:
            self.grad = np.array([0])
            self.grad_flag = False
        del mag1, mag2, xpt, ypt, day_m, month_m, year_m, hour_m, min_m, sec_m
        self.dx = self.segments[0]["dx"][0]
        self.dy = self.segments[0]["dy"][0]
        if first:
            if abs(self.x[0] - self.x[self.segments[0]["mark_samples"][-1]]) > abs(
                self.y[0] - self.y[self.segments[0]["mark_samples"][-1]]
            ):
                self.direction = 1
                self.d_lines = np.round(
                    (self.y.max() - self.y.min()) / (self.n_lines - 1), 1
                )
            else:
                self.direction = 0
                self.d_lines = np.round(
                    (self.x.max() - self.x.min()) / (self.n_lines - 1), 1
                )
        # Store original data to arrays xxx_ori
        self.sensor1_ori = np.copy(self.sensor1)
        self.sensor2_ori = np.copy(self.sensor2)
        self.grad_ori = np.copy(self.grad)
        self.line_pos = np.zeros(len(self.segments.keys()))
        if self.direction == 1:
            for key, entries in self.segments.items():
                self.line_pos[key] = entries["y"]
                self.segments[key]["pos"] = entries["y"]
        else:
            for key, entries in self.segments.items():
                self.line_pos[key] = entries["x"]
                self.segments[key]["pos"] = entries["x"]
        self.line_choice = "all"
        if self.direction == 0:
            sens = np.sign(self.y[1] - self.y[0])
            for key, entries in self.segments.items():
                i1 = entries["mark_samples"][0]
                if np.sign(self.y[i1 + 1] - self.y[i1]) == sens:
                    entries["dir"] = "odd"
                else:
                    entries["dir"] = "even"
        else:
            sens = np.sign(self.x[1] - self.x[0])
            for key, entries in self.segments.items():
                i1 = entries["mark_samples"][0]
                if np.sign(self.x[i1 + 1] - self.x[i1]) == sens:
                    entries["dir"] = "odd"
                else:
                    entries["dir"] = "even"

    def write_stn(
        self, file, data1, x, y, *, data2=None, day=None, month=None, year=None
    ):
        """
        Writes Geometrics magnetic gradiometer file in ASCII.

        Parameters
        ----------
        Obligatory parameters:
        file : str
            File where to store the data.
        data1 : numpy float array [number_of_samples_per_line, number_of_lines]
            Data of sensor 1.
        x : numpy float array [number_of_samples_per_line]
            X coordinates of all measured points.
        y : numpy float array [number_of_samples_per_line, number_of_lines]
            Y coordinates of all measured points.

        Optional parameters:
        data2 : numpy float array [number_of_samples_per_line, number_of_lines]
            Data of sensor 2. Optional. Default: np.zeros_like(data1)
        day : int
            Day of acquisition. Optional. Default: self.day[0]
        month : int
            Month of acquisition. Optional. Default: self.month[0]
        year : int
            Year of acquisition (2 ciphers). Optional. Default: self.year[0]
        direction : int
            If direction == 0: measurements were done in Y direction,
            else in X direction. Optional. Default: self.direction

        Returns
        -------
        None.

        """
        try:
            if not data1:
                print("No data given, no Geometrics stn file written")
                return
        except ValueError:
            pass
        try:
            if not data2:
                if self.grad_flag:
                    data2 = np.zeros_like(data1)
        except ValueError:
            pass
        if not day:
            day = self.day[0]
        if not month:
            month = self.month[0]
        if not year:
            year = self.year[0]
        with open(file, "w") as fh:
            fh.write("99 0 0\n")
            mark = 1
            izero = 0
            midmark = 4
            endline = 5
            startline = 36
            # Store gridded data
            if isinstance(data1, np.ndarray):
                try:
                    if not x or not y:
                        print("No coordinates given, no Geometrics stn file written")
                        return
                except ValueError:
                    pass
                ny, nx = data1.shape
                # Calculate ficticious acquisition times:
                #    Times start at 0 seconds of the day of acquisition of the first sample and
                #    increase by 0.1s between samples. Every line starts at the next full minute
                #    after the end of the former line.
                dt = ny * 0.1
                tref = np.arange(ny) * 0.1
                start = np.arange(nx) * np.ceil(dt)
                for ix, xx in enumerate(x):
                    t = start[ix] + tref
                    h = np.array(t / 3600.0, dtype=int)
                    tt = t - h * 3600
                    m = np.array(tt / 60.0, dtype=int)
                    s = tt - m * 60
                    nl = len(t)
                    fh.write(
                        f"3{xx:13.3f}{y[-1]:13.3f} {h[-1]:02d}:"
                        + f"{m[-1]:02d}:{s[-1]:05.2f} {month:02d}/"
                        + f"{day:02d}/{year:02d} {izero:11d}"
                        + f"{ix:12d}{mark:12d}{endline:5d}\n"
                    )
                    if self.grad_flag:
                        for iy in range(ny - 1, -1, -1):
                            fh.write(
                                f"0 {data1[iy,ix]:11.3f}{data2[iy,ix]:11.3f} "
                                + f"{h[iy]:02d}:{m[iy]:02d}:"
                                + f"{s[iy]:05.2f} {month:02d}/"
                                + f"{day:02d}/{year:02d}{izero:4d}\n"
                            )
                    else:
                        for iy in range(ny - 1, -1, -1):
                            fh.write(
                                f"0 {data1[iy,ix]:11.3f} "
                                + f"{h[iy]:02d}:{m[iy]:02d}:"
                                + f"{s[iy]:05.2f} {month:02d}/"
                                + f"{day:02d}/{year:02d}{izero:4d}\n"
                            )
                    fh.write(
                        f"3{xx:13.3f}{y[0]:13.3f} {h[0]:02d}:"
                        + f"{m[0]:02d}:{s[0]:05.2f} {month:02d}/"
                        + f"{day:02d}/{year:02d}{ny:11d}{ix:12d}"
                        + f"{izero:12d}{startline:5d}\n"
                    )
            # Store non-gridded data
            else:
                keys = list(data1.keys())
                nkeys = 0
                for nkeys, k in enumerate(keys):
                    if isinstance(k, str):
                        break
                for nl in range(nkeys - 1, -1, -1):
                    key = keys[nl]
                    marks = self.segments[key]["mark_samples"]
                    x = data1[key]["x"]
                    y = data1[key]["y"]
                    d1 = data1[key]["s1"]
                    if self.grad_flag:
                        d2 = data1[key]["s2"]
                    year = self.year[marks[0] : marks[-1]]
                    month = self.month[marks[0] : marks[-1]]
                    day = self.day[marks[0] : marks[-1]]
                    h = self.hour[marks[0] : marks[-1]]
                    m = self.minute[marks[0] : marks[-1]]
                    s = self.second[marks[0] : marks[-1]]
                    n1 = 0
                    n2 = 0
                    for i, j in enumerate(range(len(marks) - 1, 0, -1)):
                        if i == 0:
                            fh.write(
                                f"3{x[-1]:13.3f}{y[-1]:13.3f} {h[-1]:02d}:"
                                + f"{m[-1]:02d}:{s[-1]:05.2f} {month[-1]:02d}/"
                                + f"{day[-1]:02d}/{year[-1]:02d} {izero:11d}"
                                + f"{nl:12d}{j:12d}{endline:5d}\n"
                            )
                        else:
                            fh.write(
                                f"3{x[n1]:13.3f}{y[n1]:13.3f} {h[n1]:02d}:"
                                + f"{m[n1]:02d}:{s[n1]:05.2f} {month[n1]:02d}/"
                                + f"{day[n1]:02d}/{year[n1]:02d} {n2-n1:11d}"
                                + f"{nl:12d}{j:12d}{midmark:5d}\n"
                            )
                        n1 = marks[j - 1] - marks[0]
                        n2 = marks[j] - marks[0]
                        if self.grad_flag:
                            for iy in range(n2 - 1, n1 - 1, -1):
                                fh.write(
                                    f"0 {d1[iy]:11.3f}{d2[iy]:11.3f} "
                                    + f"{h[iy]:02d}:{m[iy]:02d}:"
                                    + f"{s[iy]:05.2f} {month[iy]:02d}/"
                                    + f"{day[iy]:02d}/{year[iy]:02d}{izero:4d}\n"
                                )
                        else:
                            for iy in range(ny - 1, n1 - 1, -1):
                                fh.write(
                                    f"0 {d1[iy]:11.3f} "
                                    + f"{h[iy]:02d}:{m[iy]:02d}:"
                                    + f"{s[iy]:05.2f} {month[iy]:02d}/"
                                    + f"{day[iy]:02d}/{year[iy]:02d}{izero:4d}\n"
                                )
                    fh.write(
                        f"3 {x[0]:13.3f}{y[0]:13.3f} {h[0]:02d}:"
                        + f"{m[0]:02d}:{s[0]:05.2f} {month[0]:02d}/"
                        + f"{day[0]:02d}/{year[0]:02d}{n2-n1:11d}{nl:12d}"
                        + f"{izero:12d}{startline:5d}\n"
                    )
        print(f"\nfile {file} written")

    def clean_data(
        self, min_fix=None, max_fix=None, percent_down=None, percent_up=None
    ):
        """
        Set data to np.nan under certain conditions which may be:

            min_fix: all data below this value are set to nan
            max_fix: all data above this value are set to nan
            percent_down: The lowermost percentile values are set to nan
                        A value of 0.01 means that all values lower than the
                        1% quantile are set to None
            percent_up: The uppermost percentile values are set to nan
                        A value of 0.01 means that all values higher than the
                        99% quantile are set to None

        The operation is done in situ.
        """
        print("\n")
        if min_fix:
            self.sensor1[self.sensor1 < min_fix] = np.nan
            if self.grad_data:
                self.sensor2[self.sensor2 < min_fix] = np.nan
            print(f"Clip below {np.round(min_fix,1)}")
        if max_fix:
            self.sensor1[self.sensor1 > max_fix] = np.nan
            if self.grad_data:
                self.sensor2[self.sensor2 > max_fix] = np.nan
            print(f"Clip above {np.round(max_fix,1)}")
        if percent_down:
            vmin1 = np.nanquantile(self.sensor1, percent_down)
            self.sensor1[self.sensor1 < vmin1] = np.nan
            if self.grad_data:
                vmin2 = np.nanquantile(self.sensor2, percent_down)
                self.sensor2[self.sensor2 < vmin2] = np.nan
                print(
                    f"Clip below {np.round(vmin1,1)} for sensor 1 and "
                    + f"{np.round(vmin2,1)} for sensor 2"
                )
            else:
                print(f"Clip below {np.round(vmin1,1)}")
        if percent_up:
            vmax1 = np.nanquantile(self.sensor1, 1 - percent_up)
            self.sensor1[self.sensor1 > vmax1] = np.nan
            if self.grad_data:
                vmax2 = np.nanquantile(self.sensor2, 1 - percent_up)
                self.sensor2[self.sensor2 > vmax2] = np.nan
                print(
                    f"Clip above {np.round(vmax1,1)} for sensor 1 and "
                    + f"{np.round(vmax2,1)} for sensor 2"
                )
            else:
                print(f"Clip above {np.round(vmax1,1)}")
        if self.grad_data:
            self.grad = (self.sensor1 - self.sensor2) / self.d_sensor

    def line_days(self):
        """
        Create a dictionary containing for every day of data acquisition the
        numbers of the lines having been measured.

        Returns
        -------
        self.lines_per_day has as key the number of the day. It is implicitely
            supposed that the days are unique, i.e. measurements were not done
            over a period longer than a month.
            For every entrance, a list contains the line numbers

        """
        days = np.unique(self.day)
        for d in days:
            self.lines_per_day[d] = []
            for key, entries in self.segments.items():
                if self.day[entries["mark_samples"][0]] == d:
                    self.lines_per_day[d].append(key)

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
        year: numpy int array
            year of acquisition of all data points extracted.
        month: numpy int array
            month of acquisition of all data points extracted.
        day: numpy int array
            day of acquisition of all data points extracted.
        hour: numpy int array
            hour of acquisition of all data points extracted.
        minute: numpy int array
            minute of acquisition of all data points extracted.
        second: numpy float array
            second of acquisition of all data points extracted.
        time: numpy int array
            second of day of acquisition of all data points extracted.

        """
        n1 = self.segments[i_line]["mark_samples"][0]
        n2 = self.segments[i_line]["mark_samples"][-1]
        if self.grad_data:
            return (
                self.sensor1[n1:n2],
                self.sensor2[n1:n2],
                self.x[n1:n2],
                self.y[n1:n2],
                self.year[n1:n2],
                self.month[n1:n2],
                self.day[n1:n2],
                self.hour[n1:n2],
                self.minute[n1:n2],
                self.second[n1:n2],
                self.time[n1:n2],
            )
        dum = np.array([0])
        return (
            self.sensor1[n1:n2],
            dum,
            self.x[n1:n2],
            self.y[n1:n2],
            self.year[n1:n2],
            self.month[n1:n2],
            self.day[n1:n2],
            self.hour[n1:n2],
            self.minute[n1:n2],
            self.second[n1:n2],
            self.time[n1:n2],
        )

    def geometrics_lines(self):
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
              "x":  E-W coordinates of data points
              "y"   N-S coordinates of data points
        """
        data = {}
        for i in range(self.n_lines):
            s1, s2, x, y, _, _, _, _, _, _, _ = self.get_line(i)
            data[i] = {}
            data[i]["s1"] = s1
            data[i]["s2"] = s2
            data[i]["x"] = x
            data[i]["y"] = y
            return data

    def get_segment(self, i_line=0, i_seg=0):
        """
        Get data of one segment (data between two marks) of a line

        Parameters
        ----------
        i_line : int, optional
            Number of line to be extracted (counting starts at 0). The default is 0.
        i_seg : int, optional
            Number of segment within line to be extracted (counting starts at 0).
            The default is 0.

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
        year: numpy int array
            year of acquisition of all data points extracted.
        month: numpy int array
            month of acquisition of all data points extracted.
        day: numpy int array
            day of acquisition of all data points extracted.
        hour: numpy int array
            hour of acquisition of all data points extracted.
        minute: numpy int array
            minute of acquisition of all data points extracted.
        second: numpy float array
            second of acquisition of all data points extracted.
        time: numpy int array
            second of day of acquisition of all data points extracted.

        """

        n1 = self.segments[i_line]["mark_samples"][i_seg]
        n2 = self.segments[i_line]["mark_samples"][i_seg + 1]
        if self.grad_data:
            return (
                self.sensor1[n1:n2],
                self.sensor2[n1:n2],
                self.x[n1:n2],
                self.y[n1:n2],
                self.year[n1:n2],
                self.month[n1:n2],
                self.day[n1:n2],
                self.hour[n1:n2],
                self.minute[n1:n2],
                self.second[n1:n2],
                self.time[n1:n2],
            )
        dum = np.array([0])
        return (
            self.sensor1[n1:n2],
            dum,
            self.x[n1:n2],
            self.y[n1:n2],
            self.year[n1:n2],
            self.month[n1:n2],
            self.day[n1:n2],
            self.hour[n1:n2],
            self.minute[n1:n2],
            self.second[n1:n2],
            self.time[n1:n2],
        )

    def read_base(self, infile, year=None):
        """
        Read data from a base station. It is supposed the base station is a
        Geometrics PPM station. The function recognizes automatically whether
        it is a G-856 (old version) or a G-857 (newer version) instrument

        If other data have already been read in, the new ones are concatenated
        to the existing data. Before leaving the subroutine, data are sorted by
        measurement time. In this way, the order of data storage within a file
        and the order of reading of different files are without importance.

        Parameters
        ----------
        infile : str
            Name of file containing base station data.
        year : int, optional
            If base station data were recorded with a G-856 PPM, the date is
            coded as Julian day without the year of acquisition. In this case,
            the year should be given. If Geometrics G-858 (Cs magnetometer) data
            have been input already, the year of those data may be used and the
            variable may be left "None". The default is None.

        Data are stored in the following arrays (usually, they are base station data):
            self.base : 1D numpy float array contains all base station data
            self.day_base   : 1D numpy int array contains day of acquisition for all data
            self.month_base : 1D numpy int array contains month of acquisition for all data
            self.year_base  : 1D numpy int array contains year of acquisition for all data
            self.hour_base  : 1D numpy int array contains hour of acquisition for all data
            self.minute_base: 1D numpy int array contains minute of acquisition for all data
            self.second_base: 1D numpy float array contains second of acquisition for all data
            self.time_base  : 1D numpy float array contains second within year for all data

        """

        if not year:
            year = self.year[0]

        with open(infile, "r") as fh:
            lines = fh.readlines()
        nlines = len(lines)
        self.year_base = list(self.year_base)
        self.month_base = list(self.month_base)
        self.day_base = list(self.day_base)
        self.jday_base = list(self.jday_base)
        self.hour_base = list(self.hour_base)
        self.minute_base = list(self.minute_base)
        self.second_base = list(self.second_base)
        self.base = list(self.base)
        self.time_base = list(self.time_base)
        if lines[0][0] == "*":
            for k in range(nlines):
                l = lines[k].split()
                try:
                    j_day = int(l[2])
                except IndexError:
                    break
                d, m = self.u.julian2date(j_day, year)
                self.jday_base.append(j_day)
                self.year_base.append(year)
                self.month_base.append(m)
                self.day_base.append(d)
                self.hour_base.append(int(l[3][0:2]))
                self.minute_base.append(int(l[3][2:4]))
                self.second_base.append(float(l[3][4:6]))
                self.time_base.append(j_day)
                if len(l) == 5:
                    self.base.append(float(l[-1].split("?")[1]) / 10.0)
                else:
                    self.base.append(float(l[-1]) / 10.0)
        else:
            for k in range(2, nlines):
                l = lines[k].split(",")
                try:
                    self.month_base.append(int(l[2][0:2]))
                except IndexError:
                    break
                self.day_base.append(int(l[2][3:5]))
                self.year_base.append(int(l[2][6:8]))
                self.hour_base.append(int(l[3][0:2]))
                self.jday_base.append(
                    self.u.date2julian(
                        self.day_base[-1], self.month_base[-1], self.year_base[-1]
                    )
                )
                self.minute_base.append(int(l[3][3:5]))
                self.second_base.append(float(l[3][6:8]))
                self.time_base.append(self.jday_base[-1])
                self.base.append(float(l[6]))
        self.month_base = np.array(self.month_base, dtype=int)
        self.day_base = np.array(self.day_base, dtype=int)
        self.year_base = np.array(self.year_base, dtype=int)
        self.hour_base = np.array(self.hour_base, dtype=int)
        self.minute_base = np.array(self.minute_base, dtype=int)
        self.second_base = np.array(self.second_base, dtype=float)
        self.base = np.array(self.base, dtype=float)
        self.time_base = (
            np.array(self.time_base) * 86400.0
            + (self.hour_base * 60.0 + self.minute_base) * 60.0
            + self.second_base
        )
        index = np.argsort(self.time_base)
        self.month_base = self.month_base[index]
        self.day_base = self.day_base[index]
        self.year_base = self.year_base[index]
        self.hour_base = self.hour_base[index]
        self.minute_base = self.minute_base[index]
        self.second_base = self.second_base[index]
        self.base = self.base[index]
        self.time_base = self.time_base[index]

    def write_base(self, outfile):
        """
        Write base station data in G-856 format.
        Parameters
        ----------
        outfile : str
            Name of file where to write the base station data.

        """
        with open(outfile, "w") as fo:
            for i, b in enumerate(self.base):
                if np.isnan(b):
                    continue
                fo.write(
                    f"*  0 {self.jday_base[i]:3d} {self.hour_base[i]:02d}"
                    + f"{self.minute_base[i]:02d}{int(self.second_base[i]):02d}"
                    + f"{i:5d}{int(b*10):7d}\n"
                )
