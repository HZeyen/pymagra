# -*- coding: utf-8 -*-
"""
Last modified on Apr 23, 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Université Paris-Saclay, France
"""

import sys
import os
from copy import deepcopy
from datetime import datetime
import numpy as np
from PyQt5 import QtWidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
from scipy import interpolate
from ..in_out.dialog import dialog
from .potential_prism import Prism_calc as PP
from . import mag_grav_utilities as utils
from ..plotting.new_window import newWindow


class inversion:
    """
    class controlling inversion of data sets

    contains the following methods:
        - __init__
        - run_inversion
        - check_regularization
        - sigmas
        - show_results2D
        - show_results3D
        - plot_2D
        - show_synthetic
        - save_model
        - get_inversion_parameters
        - get_area2D
        - get_area3D
        - prepare_data
        - set_plot_depths
        - set_prisms
        - set_prism_test
        - get_variances
        - write_parameters
    """

    def __init__(
        self,
        data_class,
        data,
        x,
        y,
        z,
        topo=None,
        earth=None,
        data_type="m",
        line_pos=None,
        direction="N",
        dim=3,
        act="I",
    ):
        """
        Initialization of inversion.

        Parameters
        ----------
        data_class : Instance of class Data
        data : list of one or two 1D or 2D numpy float arrays
            Data to be inverted [nT of mGal, depending on data_type].
            For 2.5D inversion, data are along a line in a 1D array.
            For 3D inversion, gridded data are used and passed as 2D array
            If two sensors have been used, each data set is one entry of the
            list. If only one sensor is used, the list has only one element.
        x, y, z : lists of one or two 1D or 2D numpy float arrays
            Coordinates of all data points having each the same structure as
            "data".
            If y[0] == None, 2D inversion is done and all y coordinates are
            set to 0. I.e., the coordinates along the line are X-coordinates.
        topo : 1D numpy float array or None, oprional; Default: None
            If None, topography is interpolated from gridded topography array
            data_class.topo_inter. Else, topography is used as given.
        earth: class Earth_mag object, optional. Default=None
            Contains the properties of the Earth's magnetic field
        data_type : str, optional. Default: "m"
            Type of data to be inverted: "m"=magnetic data, "g"=gravity data.
            May be capitals or not.
        line_pos : float, optional. Defuult: None
            For 2.5D inversion, coordinate of the line to be inverted (X or Y).
            Only used for plotting.
        direction : str, optional. Default: "N"
            Line direction for 2.5D inversion ("N" is parallel to Y axis,
            "E" parallel to X axis)
        act : str, optional. Default "I"
            May be "I" for inversion of "D" for direct calculation. Is used to
            define the folder where to store results.

        """
        self.data_class = data_class
        self.data_shape = data[0].shape
        self.data1_shape = data[0].shape
        self.data = data[0].flatten()
        self.line_pos = line_pos
        self.direction = direction
        self.xo1 = x[0].flatten()
        self.zo1 = -z[0].flatten()
        self.index_data = np.where(np.isfinite(self.data))[0]
        self.datao1 = np.copy(self.data)
        self.data = self.data[self.index_data]
        self.n_data = len(self.data)
        self.n_data1 = len(self.data)
        self.x = np.copy(self.xo1[self.index_data])
        self.z = np.copy(self.zo1[self.index_data])
        # If UTM (or similar) coordinates are used, coordinates are divided by 1000 for
        # plotting in order to keep axis annotation short
        if max(self.x) - min(self.x) > 10000.0:
            self.dfac = 0.001
        else:
            self.dfac = 1.0
        # If no topography is given, the one of class data is used
        if topo is None:
            self.topo = -self.data_class.topo_inter.flatten()
        # If topography is given and 2D inversion is used, extend topography vector
        # outside the data area, since else, spline interpolation may invent quite
        # huge, senseless values
        else:
            if dim == 2:
                index = np.where(np.isfinite(topo))[0]
                topo = topo[index]
                xx = np.copy(self.x)
                sort_index = np.argsort(xx)
                topo = topo[sort_index]
                xx = xx[sort_index]
                xx, indices = np.unique(xx, return_index=True)
                topo = topo[indices]
                self.topo = np.zeros(len(topo) + 10)
                self.topo[5 : len(topo) + 5] = -np.copy(topo)
                self.topo[:5] = -topo[0]
                self.topo[-5:] = -topo[-1]
                self.x_topo = np.zeros(len(xx) + 10)
                self.x_topo[5 : len(xx) + 5] = np.copy(xx)
                self.x_topo[:5] = xx[0] - (np.arange(5, 0, -1)) * 2 / self.dfac
                self.x_topo[-5:] = xx[-1] + (np.arange(5) + 1) * 2 / self.dfac
            else:
                self.topo = topo.flatten()
        index_topo = np.where(np.isfinite(self.topo))
        self.topo_flag = self.data_class.topo_flag
        self.dim = dim
        if y[0] is None:
            self.y = np.zeros_like(self.x)
            self.yo1 = np.zeros_like(self.xo1)
        else:
            self.yo1 = y[0].flatten()
            self.y = self.yo1[self.index_data]
        if self.topo_flag:
            if dim == 2:
                self.topo_inter = interpolate.interp1d(
                    self.x_topo, self.topo, fill_value="extrapolate"
                )
            else:
                self.topo_inter = interpolate.NearestNDInterpolator(
                    list(zip(self.x, self.y)), self.topo[index_topo]
                )
        # self.sensor1_data = self.n_data
        # If only one data set is to be inverted, set values of second one to None
        if len(data) == 1:
            self.data2 = None
            self.x2 = None
            self.y2 = None
            self.z2 = None
            self.n_sensor = 1
            # self.sensor2_data = 0
            self.n_data2 = 0
        # If two data sets are available, copy data and coordinates into their
        # respective arrays
        else:
            self.data2_shape = data[1].shape
            self.data2 = data[1].flatten()
            self.index_data2 = np.where(np.isfinite(self.data2))
            self.xo2 = x[1].flatten()
            self.zo2 = -z[1].flatten()
            self.x2 = np.copy(self.xo2[self.index_data2])
            self.z2 = np.copy(self.zo2[self.index_data2])
            if self.dim == 3:
                self.yo2 = y[1].flatten()
                self.y2 = np.copy(self.yo2[self.index_data2])
            else:
                self.y2 = np.zeros_like(self.x2)
                self.yo2 = np.zeros_like(self.xo2)
            self.n_sensor = 2
            self.n_data2 = len(self.data2)
        self.data_type = data_type.lower()
        self.earth = deepcopy(earth)
        if direction in ("N", "S", 0.0, 180.0):
            self.earth.dec += 90.0
            self.earth.earth_components()
        self.sus_inv = False
        self.rem_inv = False
        self.rho_inv = False
        self.positive = False
        self.log_plot_flag = False
        self.xprism_min = 0.0
        self.xprism_max = 0.0
        self.dx_prism = 0.0
        self.min_size_x = 0.0
        self.zprism_min = 0.0
        self.zprism_max = 0.0
        self.zprism_top = 1.0e6
        self.dz_prism = 0.0
        self.min_size_z = 0.0
        self.yprism_min = 0.0
        self.yprism_max = 0.0
        self.dy_prism = 0.0
        self.min_size_y = 0.0
        self.sigma_mag = 1.0
        self.sigma_grav = 1.0
        self.sigma_sus = 0.001
        self.sigma_rem = 0.04
        self.sigma_rho = 1.0
        self.iteration = 0
        self.max_iter = 10
        self.add_iter = 10
        self.max_diff_fac = 0.01
        self.max_rel_diff = 0.00001
        self.gam = 0.00001
        if "m" in data_type:
            self.lam = 0.001
        else:
            self.lam = 0.01
        self.lam_fac = 0.7
        self.lam_min = 1.0e-6
        self.gam_fac = 0.7
        self.gam_min = 1.0e-9
        self.sigma_mag = 1.0
        self.sigma_grav = 1.0
        self.sigma_sus = 1.0e-5
        self.sigma_rem = 1.0
        self.sigma_rho = 10.0
        self.depth_ref = 1.0
        self.width_max = 5
        self.max_amp = 0.1
        self.fig_sus = None
        self.fig_rem = None
        self.fig_rho = None
        self.fig_theo = None
        self.fig_theo2 = None
        self.pi4 = np.pi * 4.0
        self.param_prism = []
        self.prism_layer = []
        self.par_hist = []
        self.stop_RMS_flag = False
        self.stop_Diff_flag = False
        self.RMS_misfit = []
        self.rel_RMS_misfit = []
        self.iteration = -1
        self.prism_del = {}
        self.prism_new = {}
        self.fig_follow = []
        now = datetime.now()
        self.c_time = now.strftime("%H-%M-%S")
        self.d1 = now.strftime("%Y-%m-%d")
        if act == "I":
            self.folder = f"inversion_{self.d1}_{self.c_time}"
        else:
            self.folder = f"direct_{self.d1}_{self.c_time}"
        os.makedirs(self.folder)

    def run_inversion(self):
        """
        Do 2D or 3D inversion using a self-refining algorithm

        Returns
        -------
        None.

        """
        self.n_data = len(self.data)
        std_data = np.std(self.data)
        if self.iteration > 0:
            self.iteration -= 1
        i0 = 0
        if self.dim == 2:
            prism_add = 3
        else:
            prism_add = 7
        # Start inversion
        while True:
            self.iteration += 1
            if self.max_iter == 0:
                print("\nOnly forward model calculated")
                self.G = self.mPrism.create_Frechet(
                    self.sus_inv, self.rem_inv, self.rho_inv, self.x, self.y, self.z
                )
                self.data_mod = np.matmul(self.G, self.params)
                return
            # Calculate effect of actual model
            print(f"\nStart iteration {self.iteration}")

            # Create Frechet matrix and covariance matrices
            self.n_param = self.mPrism.get_n_param(
                self.sus_inv, self.rem_inv, self.rho_inv
            )
            self.S = self.mPrism.create_smooth(
                self.sus_inv,
                self.rem_inv,
                self.rho_inv,
                self.sigma_sus,
                self.sigma_rem,
                self.sigma_rho,
                self.depth_ref,
            )
            self.G = self.mPrism.create_Frechet(
                self.sus_inv, self.rem_inv, self.rho_inv, self.x, self.y, self.z
            )
            # Define data covariance and regularization matrices.
            # Since both matrices have only values on their diagonal, values are stored as
            # 1D vector. For the regularization matrix, the base value is the squared one
            # given by the user interactively as parameter variability. This value is
            # modified by the depth using as formula:
            #    factor = (average_depth_of_prism/depth_ref) squared of gravity data and
            #    to the power of three for magnetic data. Depth_ref is given interactively
            #    by the user
            self.dat = np.copy(self.data)
            self.sigma_data, self.sigma_param = self.sigmas()
            print(f"Frechet calculated, shape: {self.G.shape}")
            # Do inversion with positivity constraint
            if self.positive:
                fit = []
                self.dat = self.data_ori
                dd = np.copy(self.dat)
                self.params[:-1] = 0.001
                self.params[-1] = 0.0
                # Iterate to get best fit. Limit number of iterations to maximum 50, but stop
                # iterations if misfit does no longer decrease
                for it in range(50):
                    self.yparam = np.copy(self.params)
                    self.yparam[:-1] = np.log(self.params[:-1])
                    fac = np.copy(self.params)
                    fac[-1] = 1.0
                    G0 = self.G * fac
                    GCT = G0.T * self.sigma_data
                    G_inv = np.matmul(GCT, G0)
                    Gi_max = abs(G_inv).max()
                    Gp = self.lam * self.sigma_param * 0.001
                    Gp_max = abs(Gp).max()
                    Gp_fac = Gi_max / Gp_max
                    Gs = self.gam * self.S * 0.001
                    Gs_max = Gs.max()
                    Gs_fac = Gi_max / Gs_max
                    if self.iteration == 0 and it == 0:
                        print(f"\nMaximum GT*Cd*G: {Gi_max:0.1f}")
                        print(
                            f"   Maximum regularization: {Gp_max:0.1f} (fac:"
                            + f" {Gp_fac})"
                        )
                        print(
                            f"   Maximum smoothing     : {Gs_max:0.1f} (fac:"
                            + f" {Gs_fac})"
                        )
                        if Gp_fac > 10 or Gs_fac > 10 or Gp_fac < 0.1 or Gs_fac < 0.1:
                            self.check_regularization(Gp_fac, Gs_fac)
                    G_inv += Gs
                    G_inv[np.diag_indices(G_inv.shape[0])] += Gp
                    d_par = np.matmul(np.matmul(np.linalg.inv(G_inv), GCT), dd)
                    yp = self.yparam + d_par
                    if yp[:-1].max() > 0.0:
                        ip = np.argmax(yp[:-1])
                        factor = -self.yparam[ip] / d_par[ip]
                    else:
                        factor = 1.0
                    self.yparam[:-1] += d_par[:-1] * factor
                    self.yparam[-1] += d_par[-1]
                    self.params = np.copy(self.yparam)
                    self.params[:-1] = np.exp(self.yparam[:-1])
                    self.params[:-1][self.params[:-1] < 1.0e-7] = 1.0e-7
                    # At the first iteration, if one starts with several layers, I observed often
                    # that the averages of the parameters in each layer are very different, no idea
                    # why. This is why I decided to reduce the minimum in each layer to near zero
                    # if the positivity constraint is activated
                    nlay = 0
                    zp = np.zeros(len(self.mPrism.prisms.keys()))
                    for i, key in enumerate(self.mPrism.prisms.keys()):
                        zp[i] = np.mean(self.mPrism.prisms[key].z)
                    if self.topo_flag:
                        n = np.where(zp < self.zprism_min)[0]
                        if len(n) > 0:
                            amin = self.params.max()
                            for i in n:
                                amin = min(self.params[i], amin)
                            for i in n:
                                self.params[i] -= amin - 1.0e-6
                    for i, zz in enumerate(self.z_prism[:-1]):
                        n = np.where((zp > zz) & (zp < self.z_prism[i + 1]))[0]
                        if len(n) > 0:
                            nlay += 1
                            amin = self.params.max()
                            for i in n:
                                amin = min(self.params[i], amin)
                            for i in n:
                                self.params[i] -= amin - 1.0e-6
                    self.data_mod = np.matmul(self.G, self.params)
                    dd = self.dat - self.data_mod
                    fit.append(np.nanstd(dd))
                    if it < 2:
                        continue
                    if abs((fit[-2] - fit[-1]) / fit[-2]) < 1.0e-5:
                        break
            # Do inversion without positivity constraint
            else:
                GCT = self.G.T * self.sigma_data
                G_inv = np.matmul(GCT, self.G)
                # For first iteration test whether regularization and smoothing matrices have
                # an appreciable effect. If not, give a warning message
                if self.iteration == 0:
                    mG = abs(G_inv).max()
                    print(f"\nMaximum GT*Cd*G: {mG:0.1f}")
                    mSig = (self.lam * self.sigma_param).max()
                    mSmo = (self.gam * self.S).max()
                    G_Sig = mG / mSig
                    G_Smo = mG / mSmo
                    print(
                        f"   Maximum regularization: {mSig:0.1f} (fac: " + f"{G_Sig})"
                    )
                    print(f"   Maximum smoothing: {mSmo:0.1f} (fac: " + f"{G_Smo})")
                    if G_Sig > 10 or G_Smo > 10 or G_Sig < 0.1 or G_Smo < 0.1:
                        self.check_regularization(G_Sig, G_Smo)
                G_inv[np.diag_indices(G_inv.shape[0])] += self.lam * self.sigma_param
                G_inv += self.gam * self.S
                d_par = np.matmul(np.matmul(np.linalg.inv(G_inv), GCT), self.dat)
                self.params += d_par
                # At the first iteration, if one starts with several layers, I observed often
                # that the averages of the parameters in each layer are very different, no idea
                # why. This is why I decided to eliminate the averaged in each layer if the
                # positivity constraint is not activated
                # if self.iteration == 0 and self.mod_zshape > 1:
                nlay = 0
                zp = np.zeros(len(self.mPrism.prisms.keys()))
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    zp[i] = np.mean(self.mPrism.prisms[key].z)
                if self.topo_flag:
                    n = np.where(zp < self.zprism_min)[0]
                    if len(n) > 0:
                        av = 0.0
                        for i in n:
                            av += self.params[i]
                        av /= len(n)
                        for i in n:
                            self.params[i] -= av
                        print(
                            "\nAverage parameter of topography layer: " + f"{av:0.6f}"
                        )
                for i, zz in enumerate(self.z_prism[:-1]):
                    n = np.where((zp > zz) & (zp < self.z_prism[i + 1]))[0]
                    if len(n) > 0:
                        nlay += 1
                        av = 0.0
                        for i in n:
                            av += self.params[i]
                        av /= len(n)
                        for i in n:
                            self.params[i] -= av
                        print(f"Average parameter of layer {nlay}: " + f"{av:0.6f}\nn")
            self.data_mod = np.matmul(self.G, self.params)
            # Extract new prism properties from parameter vector, set the correspondig
            #   values in the prism parameters and copy them into vector par_hist
            i0 = 0
            ss = []
            if self.sus_inv:
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].setsus(self.params[i])
                    ss.append(self.mPrism.prisms[key].getsus())
                i0 += self.mPrism.n_prisms
            if self.rem_inv:
                sr = []
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].rem = self.params[i + i0]
                    sr.append(self.mPrism.prisms[key].rem)
            if self.sus_inv:
                if self.rem_inv:
                    for i, s in enumerate(ss):
                        self.par_hist.append(np.array([s, sr[i]]))
                else:
                    self.par_hist.append(np.array(ss))
            elif self.rem_inv:
                self.par_hist.append(sr)
            if self.rho_inv:
                s = []
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].rho = self.params[i]
                    s.append(self.mPrism.prisms[key].rho)
                self.par_hist.append(np.array(s))
            # Calculate magnetic misfits and some statistics
            data = self.data_ori - self.data_mod
            data -= np.nanmedian(data)
            self.data = np.copy(data)
            std_data0 = std_data
            std_data = np.nanstd(data)
            self.std_data_rel = std_data / self.std_data_ori
            std_data_diff = (std_data0 - std_data) / self.std_data_ori
            self.RMS_misfit.append(std_data)
            self.rel_RMS_misfit.append(self.std_data_rel * 100.0)
            if self.std_data_rel < self.max_diff_fac:
                self.stop_RMS_flag = True
            if abs(std_data_diff) < self.max_rel_diff:
                self.stop_Diff_flag = True
            print(f"Maximum data difference data: {np.nanmax(abs(data))}")
            print(f"Relative Std misfit [%]: {self.std_data_rel*100}")
            print(f"Relative variation of misfit [%]: {std_data_diff*100}")
            if self.sus_inv:
                print(
                    f"Average: {self.params[-1]:0.2f}, "
                    + f"sus*10**-3: ({self.params[:-1].min()*1000.:0.3f}, "
                    + f"{self.params[:-1].max()*1000.:0.3f})\n"
                )
            elif self.rem_inv:
                print(
                    f"Average: {self.params[-1]:0.2f}, "
                    + f"rem: ({self.params[:-1].min():0.3f}, "
                    + f"{self.params[:-1].max():0.3f})\n"
                )
            else:
                print(
                    f"Average: {self.params[-1]:0.2f}, "
                    + f"rho: ({self.params[:-1].min():0.3f}, "
                    + f"{self.params[:-1].max():0.3f})\n"
                )
            # If Maximum iteration number is reached, misfit does not become smaller or
            # relative misfits are smaller than predefined minimum value, stop iterations
            if self.iteration == self.max_iter:
                print("\nMaximum number of iterations reached\n" + "Iterations stop")
                return
            if self.stop_RMS_flag:
                print("\nMisfit limit reached\nIterations stop")
                return
            if self.stop_Diff_flag:
                print("\nNo more misfit reduction.")
            # Modify prism size in area of maximum misfit
            key_m = []
            key_r = []
            self.stop_Diff_flag = False
            # In the following lines, only the mPrism.get_max_prisms lines are important.
            # The lineswith utils.get_extremes are only for testing purposes, such as the
            # plots later on.
            # if self.n_sensor == 2:
            #     d = np.copy(self.datao1)
            #     d[self.index_data] = abs(data[:self.n_data1])
            #     _, maxima1 = utils.get_extremes(
            #         d.reshape(self.data1_shape), self.width_max)
            #     d = np.copy(self.datao2)
            #     d[self.index_data2] = abs(data[self.n_data1:])
            #     _, maxima2 = utils.get_extremes(
            #         d.reshape(self.data2_shape), self.width_max)
            #     maxima = maxima1+maxima2
            # else:
            #     d = np.copy(self.datao1)
            #     d[self.index_data] = abs(data)
            #     _, maxima = utils.get_extremes(
            #         d.reshape(self.data_shape), self.width_max)
            # for pos in maxima:
            #     if len(pos) == 1:
            #         i = pos[0]
            #     else:
            #         i = pos[0]*self.data_shape[1] + pos[1]
            key_r = []
            if self.n_sensor == 2:
                d = np.copy(self.datao1)
                d[self.index_data] = abs(data[: self.n_data1])
                key_m1 = self.mPrism.get_max_prisms(
                    d.reshape(self.data1_shape),
                    self.G[: self.n_data1, : self.n_prisms],
                    self.index_data,
                    max_lim=self.max_amp,
                    width=self.width_max,
                )
                d = np.copy(self.datao2)
                d[self.index_data2] = abs(data[self.n_data1 :])
                key_m2 = self.mPrism.get_max_prisms(
                    d.reshape(self.data2_shape),
                    self.G[self.n_data1 :, : self.n_prisms],
                    self.index_data2,
                    max_lim=self.max_amp,
                    width=self.width_max,
                )
                key_m = list(np.unique(np.array(key_m1 + key_m2)))
            else:
                d = np.copy(self.datao1)
                d[self.index_data] = abs(data)
                key_m = self.mPrism.get_max_prisms(
                    d.reshape(self.data_shape),
                    self.G[: self.n_data1, : self.n_prisms],
                    self.index_data,
                    width=self.width_max,
                )
            if self.sus_inv and self.rem_inv:
                if self.n_sensor == 2:
                    d = np.copy(self.datao1)
                    d[self.index_data] = abs(data[: self.n_data1])
                    key_r1 = self.mPrism.get_max_prisms(
                        d.reshape(self.data1_shape),
                        self.G[: self.n_data1, self.n_prisms :],
                        self.index_data,
                        max_lim=self.max_amp,
                        width=self.width_max,
                    )
                    d = np.copy(self.datao2)
                    d[self.index_data2] = abs(data[self.n_data1 :])
                    key_r2 = self.mPrism.get_max_prisms(
                        d.reshape(self.data2_shape),
                        self.G[self.n_data1 :, self.n_prisms :],
                        self.index_data2,
                        max_lim=self.max_amp,
                        width=self.width_max,
                    )
                    key_r = list(np.unique(np.array(key_r1 + key_r2)))
                else:
                    d = np.copy(self.datao1)
                    d[self.index_data] = abs(data)
                    key_r = self.mPrism.get_max_prisms(
                        d.reshape(self.data1_shape),
                        self.G[: self.n_data1, self.n_prisms :],
                        self.index_data,
                        max_lim=self.max_amp,
                        width=self.width_max,
                    )
            key_split = list(np.unique(np.array(key_m + key_r)))
            # Test whether prisms are marked for splitting
            if len(key_split) == 0:
                print("  Prisms reached size limit\nIteration stops")
                return
            # Test whether number of prisms becomes larger than number of data points after
            # splitting Prisms will be split into up to 8 pieces, the prism itself is
            # eliminated. Therefore the factor 7 in the next line.
            if self.n_prisms + prism_add * len(key_split) > self.n_data:
                print("Inversion stopped: More prisms than data points")
                return
            self.prism_del[self.iteration] = {}
            self.prism_new[self.iteration] = {}
            print(f"  Split prisms {key_split}")
            for _, key in enumerate(key_split):
                self.prism_del[self.iteration][key] = deepcopy(self.mPrism.prisms[key])
                key_add = self.mPrism.split(key)
                self.prism_new[self.iteration][key] = {}
                for k in key_add:
                    self.prism_new[self.iteration][key][k] = deepcopy(
                        self.mPrism.prisms[k]
                    )
            self.n_prisms = len(list(self.mPrism.prisms.keys()))
            self.params = []
            param_prism = []
            if self.sus_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.getsus())
                    param_prism.append(key)
            if self.rem_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.rem)
                    param_prism.append(key)
            if self.rho_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.rho)
                    param_prism.append(key)
            self.params.append(0.0)
            self.params = np.array(self.params)
            self.lam = max(self.lam * self.lam_fac, self.lam_min)
            self.gam = max(self.gam * self.gam_fac, self.gam_min)
            print(f"New lambda: {self.lam}; new gamma: {self.gam}")
            if self.max_iter == 0:
                return

    def check_regularization(self, G_Sig, G_Smo):
        if G_Sig > 10.0:
            if G_Smo > 10.0:
                text = (
                    "Regularization and smoothing much smaller than "
                    + "Frechet, will not influence inversion:\n"
                    + f"     Frechet/Regularization: {G_Sig:0.0f}\n"
                    + f"     Frechet/Smoothing: {G_Smo:0.0f}\n"
                    + "You may increase lambda/gamma or restart and "
                    + "decrease parameter variances\n\n"
                )
            elif G_Smo < 0.1:
                text = (
                    "Regularization much smaller than Frechet, "
                    + "will not influence inversion:\n"
                    + f"     Frechet/Regularization: {G_Sig:0.0f}\n"
                    + "\nSmoothing much larger than Frechet, too strong "
                    + "influence:"
                    + f"     Smoothing/Frechet: {1./G_Smo:0.0f}\n"
                    + "You may increase lambda and/or decrease gamma or "
                    + "restart and decrease parameter variances\n\n"
                )
            else:
                text = (
                    "Regularization much smaller than Frechet, "
                    + "Frechet, will not influence inversion:\n"
                    + f"     Frechet/Regularization: {G_Sig:0.0f}\n"
                    + "You may increase lambda or "
                    + "restart and change parameter variances\n\n"
                )
        elif G_Sig < 0.1:
            if G_Smo < 0.1:
                text = (
                    "Regularization and smoothing much larger than "
                    + "Frechet, will influence inversion too much:\n"
                    + f"     Regularization/Frechet: {1./G_Sig:0.0f}\n"
                    + f"     Smoothing/Frechet: {1./G_Smo:0.0f}\n"
                    + "You may decrease lambda/gamma or restart and "
                    + "increase parameter variances\n\n"
                )
            elif G_Smo > 10.0:
                text = (
                    "Regularization much larger than Frechet, "
                    + "Frechet, will influence inversion too much:\n"
                    + f"     Regularization/Frechet: {1./G_Sig:0.0f}\n"
                    + "\nSmoothing much smaller than Frechet, no "
                    + "influence on inversion:"
                    + f"     Frechet/Smoothing: {G_Smo:0.0f}\n"
                    + "You may decrease lambda and/or increase gamma or "
                    + "restart and change parameter variances\n\n"
                )
            else:
                text = (
                    "Regularization much larger than Frechet, "
                    + "will influence inversion too much:\n"
                    + f"     Regularization/Frechet: {1./G_Sig:0.0f}\n"
                    + "You may decrease lambda or "
                    + "restart and change parameter variances\n\n"
                )
        else:
            if G_Smo > 10.0:
                text = (
                    "Smoothing much smaller than Frechet, no "
                    + "influence on inversion:"
                    + f"     Frechet/Smoothing: {G_Smo:0.0f}\n"
                    + "You may increase gamma or "
                    + "restart and change parameter variances\n\n"
                )
            if G_Smo < 0.1:
                text = (
                    "Smoothing much larger than Frechet, too much "
                    + "influence on inversion:"
                    + f"     Smoothing/Frechet: {1./G_Smo:0.0f}\n"
                    + "You may decrease gamma or "
                    + "restart and change parameter variances\n\n"
                )
        answer = QtWidgets.QMessageBox.warning(
            None,
            "Warning",
            text
            + "Ignore to continue nevertheless\n"
            + "Retry to give new initial lambda/gamma\n"
            + "Abort to stop inversion\n",
            QtWidgets.QMessageBox.Ignore
            | QtWidgets.QMessageBox.Abort
            | QtWidgets.QMessageBox.Retry,
            QtWidgets.QMessageBox.Ignore,
        )
        if answer == QtWidgets.QMessageBox.Abort:
            sys.exit()
        elif answer == QtWidgets.QMessageBox.Retry:
            results, okButton = dialog(
                [
                    "New Initial lambda (regularization)",
                    "New initial gamma (smoothing)",
                ],
                ["e", "e"],
                [self.lam, self.gam],
                "New inversion parameters",
            )
            if okButton:
                self.lam = float(results[0])
                self.gam = float(results[1])

    def sigmas(self):
        """
        Calculate data covariance and regularization matrices.

        Since both matrices have only values on their diagonal, values are
        stored 1D vector. For the regularization matrix, the base value is the
        squared one given by the user interactively as parameter variability.
        This value is modified by the depth using as formula:
           factor = (average_depth_of_prism/depth_ref) squared for gravity data
           and to the power of three for magnetic data. Depth_ref is given
           interactively by the user

        Returns
        -------
        sigma_data : numpy float vector (length: number of data points)
            Contains values of the diagonal of the data covariance matrix.
        sigma_param : numpy float vector (length: number of model parameters)
            Contains values of the diagonal of the data covariance matrix.

        """
        icol = 0
        fac = (self.depth_ref - 1.0) / (self.zprism_max - self.zprism_top)
        if "m" in self.data_type:
            sigma_data = np.ones(self.n_data) / self.sigma_mag**2
            i = -1
            if self.sus_inv:
                sigma_param = np.ones(self.mPrism.n_prisms) / self.sigma_sus**2
                if not np.isclose(self.depth_ref, 1.0):
                    for _, val in self.mPrism.prisms.items():
                        i += 1
                        f = 1.0 + ((val.z[0] + val.z[1]) / 2.0 - self.zprism_top) * fac
                        sigma_param[i] *= f**3
                icol += self.n_prisms
            elif self.rem_inv:
                sigma_param = np.ones(self.mPrism.n_prisms) / self.sigma_rem**2
                if not np.isclose(self.depth_ref, 1.0):
                    for _, val in self.mPrism.prisms.items():
                        i += 1
                        f = 1.0 + ((val.z[0] + val.z[1]) / 2.0 - self.zprism_top) * fac
                        sigma_param[i] *= f**3
                icol += self.n_prisms
        else:
            sigma_data = np.ones(self.n_data) / self.sigma_grav**2
            sigma_param = np.ones(self.mPrism.n_prisms) / self.sigma_rho**2
            i = -1
            if not np.isclose(self.depth_ref, 1.0):
                for _, val in self.mPrism.prisms.items():
                    i += 1
                    f = 1.0 + ((val.z[0] + val.z[1]) / 2.0 - self.zprism_top) * fac
                    sigma_param[i] *= f**2
        sigma_param = np.concatenate((sigma_param, np.array([0.0])))
        return sigma_data, sigma_param

    def show_results2D(self, file):
        """
        Plot results of 2D inversion.

        Parameters
        ----------
        file : str
            Base name of file where to store the image

        """
        self.equal_flag = False
        title = self.data_class.title
        if self.sus_inv:
            txt = "Susceptibility"
        elif self.rem_inv:
            txt = "Remanence"
        else:
            txt = "Density"
        # Plot misfit evolution
        if self.iteration > 0:
            self.fig_mis2 = newWindow(f"{txt} model", 1000, 750, 15, 15)
            self.fig_mis2.fig.tight_layout(w_pad=15, h_pad=2)
            ax_mis = self.fig_mis2.fig.add_subplot()
            ax_mis.plot(np.arange(len(self.rel_RMS_misfit)), self.rel_RMS_misfit)
            ax_mis.set_title(title, fontsize=14)
            ax_mis.set_xlabel("Iteration #", fontsize=12)
            ax_mis.set_ylabel("Relative RMS misfit [%]", fontsize=12)
            ax_mis.set_xlim([0, len(self.RMS_misfit) - 1])
            self.fig_mis2.show()
            fil = os.path.join(self.folder, file + "_misfit.png")
            self.fig_mis2.fig.savefig(fil)

        # Plot measured and calculated data
        while True:
            self.plot_2D(file, inv_flag=True)
            # Save inversion control parameters to file parameters.dat
            file_name = os.path.join(self.folder, "parameters.dat")
            self.write_parameters(file_name)
            if self.equal_flag:
                fil = os.path.join(self.folder, file + "_scaled.png")
            else:
                fil = os.path.join(self.folder, file + "_not-scaled.png")
            self.fig_2D.fig.savefig(fil)
            while True:
                event = self.fig_2D.get_event()
                if self.iteration > 0:
                    self.fig_mis2.close_window()
                if event.name == "key_press_event":
                    self.fig_2D.close_window()
                    if event.key == "enter":
                        return False
                    if event.key in ("e", "E"):
                        self.equal_flag = not self.equal_flag
                        break
                    elif event.key in ("r", "R"):
                        return True
                    elif event.key in ("c", "C"):
                        self.log_plot_flag = not self.log_plot_flag
                        break

    def show_results3D(self):
        """
        Plot inversion results
        - maps of prism parameters (susceptibilities, remanences of densities)
        - maps of calculated data and difference calculated minus measured
        - evolution of misfit

        Returns
        -------
        None.

        """
        self.set_plot_depths()
        if self.sus_inv:
            txt = "Sus"
            c_txt = "Sus [SI*10**-3]"
        elif self.rem_inv:
            txt = "Rem"
            c_txt = "Remanence [A/m]"
        else:
            txt = "Density"
            c_txt = "Density [kg/m3]"
        # Calculate number of subplots to be done, depending on number of layers
        #    calculated
        i0 = 0
        n_prop_plots = len(self.nz_plot)
        max_ax_col = int(np.ceil(np.sqrt(n_prop_plots)))
        max_ax_row = int(np.ceil(n_prop_plots / max_ax_col))
        nax_plot = n_prop_plots - max_ax_col
        npltx = min(max_ax_col, n_prop_plots)
        nplty = max_ax_row
        width = npltx * 5
        height = nplty * 5
        par = []
        # Plot model parameter distribution
        for key, val in self.mPrism.prisms.items():
            if self.sus_inv:
                par.append(val.getsus() * 1000.0)
            elif self.rem_inv:
                par.append(val.rem)
            else:
                par.append(val.rho)
        par = np.array(par)
        i0 += self.mPrism.n_prisms
        k = 0
        vmin = np.nanquantile(par, 0.01)
        vmax = np.nanquantile(par, 0.99)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.fig_par = newWindow(f"{txt} model", 2400, 1800, width, height)
        self.fig_par.fig.tight_layout(w_pad=15, h_pad=2)
        self.gs = GridSpec(nplty * 10, npltx * 10 + 2, self.fig_par.fig)
        figy0 = -10
        for i in range(nplty):
            figy0 += 10
            figx0 = -10
            for j in range(npltx):
                figx0 += 10
                patches = []
                col = []
                ax = self.fig_par.fig.add_subplot(
                    self.gs[figy0 : figy0 + 8, figx0 : figx0 + 8]
                )
                if k >= n_prop_plots:
                    ax.axis("off")
                    k += 1
                    continue
                for key, val in self.mPrism.prisms.items():
                    if val.z.min() <= self.z_plot[k] <= val.z.max():
                        if self.xprism_max - self.xprism_min > 10000.0:
                            x1 = val.x[0] / 1000.0
                            x2 = val.x[1] / 1000.0
                            y1 = val.y[0] / 1000.0
                            y2 = val.y[1] / 1000.0
                        else:
                            x1 = val.x[0]
                            x2 = val.x[1]
                            y1 = val.y[0]
                            y2 = val.y[1]
                        if self.sus_inv:
                            col.append(val.getsus() * 1000.0)
                        elif self.rem_inv:
                            col.append(val.rem)
                        else:
                            col.append(val.rho)
                        patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1))
                col = np.array(col)
                p = PatchCollection(
                    patches, cmap="rainbow", norm=norm, edgecolors=("black",)
                )
                p.set_array(col)
                ax.add_collection(p)
                ax.set_title(f"{txt} at {self.z_plot[k]:0.1f} m", fontsize=10)
                ax.tick_params(top=True, right=True, labelsize=10)
                if k + npltx + 1 > n_prop_plots:
                    ax.set_xlabel(f"Easting [{self.ax_unit}]", fontsize=10)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(f"Northing [{self.ax_unit}]", fontsize=10)
                elif j == npltx - 1 or k == n_prop_plots - 1:
                    ax.yaxis.tick_right()
                else:
                    ax.set_yticklabels([])
                ax.set_xlim([self.xprism_min * self.dfac, self.xprism_max * self.dfac])
                ax.set_ylim([self.yprism_min * self.dfac, self.yprism_max * self.dfac])
                ax.set_aspect("equal", adjustable="box")
                if k < nax_plot:
                    ax.set_xlabel("")
                if j > 0:
                    ax.set_ylabel("")
                k += 1
        ax = self.fig_par.fig.add_subplot(self.gs[1:-1, -1:])
        ax.axis("off")
        cax = ax.inset_axes([0.0, 0.0, 0.9, 0.9], transform=ax.transAxes)
        cbar = plt.colorbar(p, orientation="vertical", cax=cax, fraction=0.9)
        cbar.set_label(c_txt, fontsize=10)
        cbar.ax.tick_params(labelsize=10)

        self.fig_par.setHelp("Press ENTER to finish, R to continue iterations")
        self.fig_par.show()
        self.write_parameters(os.path.join(self.folder, "parameters.dat"))
        self.fig_par.fig.savefig(os.path.join(self.folder, f"{txt}_distribution.png"))

        if "m" in self.data_type:
            self.fig_theo = newWindow("Magnetic data", 1500, 1000, 20, 15)
            self.fig_theo.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Mag_sensor1_calc&diff.png")
            title = "Modelled magnetic sensor1"
            unit = "nT"
            if self.iteration > 0:
                title2 = (
                    "Mag difference sensor1\nrel. RMS misfit: "
                    + f"{self.std_data_rel*100:0.2f}%"
                )
            else:
                title2 = "Mag difference sensor1"
        else:
            self.fig_theo = newWindow("Gravity data", 1500, 1000, 20, 15)
            self.fig_theo.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Gravi_calc&diff.png")
            title = "Modelled gravity data"
            unit = "mGal"
            if self.iteration > 0:
                title2 = (
                    "Gravity difference\nrel. RMS misfit: "
                    + f"{self.std_data_rel*100:0.2f}%"
                )
            else:
                title2 = "Gravity difference"
        self.ax_theo = []
        ddx = self.xplt_max - self.xplt_min
        ddy = self.yplt_max - self.yplt_min
        if ddx > 1.5 * ddy:
            self.gs = GridSpec(18, 10, self.fig_theo)
            self.ax_theo.append(self.fig_theo.fig.add_subplot(self.gs[1:7, 1:]))
            self.ax_theo.append(self.fig_theo.fig.add_subplot(self.gs[11:17, 1:]))
            self.bar_or = "vertical"
            #                    anchor = 'E'
            self.nticks = 10
        # Horizontal layout
        else:
            self.gs = GridSpec(10, 18, self.fig_theo)
            self.ax_theo.append(self.fig_theo.fig.add_subplot(self.gs[1:, 1:7]))
            self.ax_theo.append(self.fig_theo.fig.add_subplot(self.gs[1:, 11:17]))
            self.bar_or = "horizontal"
            #                    anchor = 'S'
            self.nticks = 5

        # Plot magnetic anomalies produced by inverted model
        data = self.datao1.flatten()
        data[self.index_data] = self.data_mod[: self.n_data1]
        data = data.reshape(self.data1_shape)
        med = np.nanmedian(data)
        data -= med
        vmin = np.ceil(np.nanquantile(data, 0.005) * 1000) / 1000
        vmax = np.ceil(np.nanquantile(data, 0.995) * 1000) / 1000
        br_map, norm = utils.mag_color_map(vmin, vmax)
        im, cbar = utils.data_plot(
            data,
            self.fig_theo.fig,
            self.ax_theo[0],
            title=f"{title}\nMedian: {med:0.1f}",
            xtitle=f"Easting [{self.ax_unit}]",
            ytitle=f"Northing [{self.ax_unit}]",
            cmap=br_map,
            norm=norm,
            cbar_title=unit,
            extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
        )
        # plot prism contours
        for key, val in self.mPrism.prisms.items():
            x1 = val.x[0]
            x2 = val.x[1]
            y1 = val.y[0]
            y2 = val.y[1]
            self.ax_theo[0].plot(
                [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k", linewidth=1
            )
        self.ax_theo[0].set_xlim([self.xplt_min, self.xplt_max])
        self.ax_theo[0].set_ylim([self.yplt_min, self.yplt_max])
        self.ax_theo[0].grid(visible=True, which="both")
        self.ax_theo[0].set_xlabel("")
        # Plot misfit
        data = self.datao1.flatten()
        data[self.index_data] = (
            self.data_mod[: self.n_data1] - self.data_ori[: self.n_data1]
        )
        data -= np.nanmedian(data)
        data = data.reshape(self.data1_shape)
        vmin = np.ceil(np.nanquantile(data, 0.005) * 1000) / 1000
        vmax = np.floor(np.nanquantile(data, 0.995) * 1000) / 1000
        br_map, norm = utils.mag_color_map(vmin, vmax)
        im, cbar = utils.data_plot(
            data,
            self.fig_theo.fig,
            self.ax_theo[1],
            title=f"{title2}",
            xtitle=f"Easting [{self.ax_unit}]",
            ytitle=f"Northing [{self.ax_unit}]",
            extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
            cmap=br_map,
            cbar_title=f"Difference calc-meas [{unit}]",
            norm=norm,
        )
        self.ax_theo[1].grid(visible=True, which="both")
        self.fig_theo.show()
        self.fig_theo.fig.savefig(plt_file)

        # Plot results for sensor 2
        if self.n_sensor == 2:
            self.fig_theo2 = newWindow("Magnetic data", 1500, 1000, 24, 18)
            self.fig_theo2.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file2 = os.path.join(self.folder, "Mag_sensor2_calc&diff.png")
            title2 = (
                "Mag difference sensor2, rel. RMS misfit: "
                + f"{self.std_data_rel*100:0.2f}%"
            )
            self.ax_theo2 = []
            if ddx > 1.5 * ddy:
                self.gs = GridSpec(18, 10, self.fig_theo2)
                self.ax_theo2.append(self.fig_theo2.fig.add_subplot(self.gs[1:7, 1:]))
                self.ax_theo2.append(self.fig_theo2.fig.add_subplot(self.gs[11:17, 1:]))
                self.bar_or = "vertical"
                #                    anchor = 'E'
                self.nticks = 10
            # Horizontal layout
            else:
                self.gs = GridSpec(10, 18, self.fig_theo2)
                self.ax_theo2.append(self.fig_theo2.fig.add_subplot(self.gs[1:, 1:7]))
                self.ax_theo2.append(self.fig_theo2.fig.add_subplot(self.gs[1:, 11:17]))
                self.bar_or = "horizontal"
                #                    anchor = 'S'
                self.nticks = 5
            # Plot magnetic anomalies produced by inverted model
            data = self.datao2.flatten()
            data[self.index_data2] = self.data_mod[self.n_data1 :]
            data = data.reshape(self.data2_shape)
            med = np.median(data)
            data -= med
            vmin = np.ceil(np.nanquantile(data, 0.005) * 1000) / 1000
            vmax = np.ceil(np.nanquantile(data, 0.995) * 1000) / 1000
            br_map, norm = utils.mag_color_map(vmin, vmax)
            im, cbar = utils.data_plot(
                data,
                self.fig_theo2.fig,
                self.ax_theo2[0],
                title=f"Modelled magnetic sensor2\nMedian: {med:0.1f}",
                xtitle=f"Easting [{self.ax_unit}]",
                ytitle=f"Northing [{self.ax_unit}]",
                cmap=br_map,
                norm=norm,
                cbar_title=unit,
                extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
            )
            # plot prism contours
            for key, val in self.mPrism.prisms.items():
                x1 = val.x[0]
                x2 = val.x[1]
                y1 = val.y[0]
                y2 = val.y[1]
                self.ax_theo2[0].plot(
                    [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k", linewidth=1
                )
            self.ax_theo2[0].set_xlim([self.xplt_min, self.xplt_max])
            self.ax_theo2[0].set_ylim([self.yplt_min, self.yplt_max])
            self.ax_theo2[0].grid(visible=True, which="both")
            self.ax_theo2[0].set_xlabel("")

            # Plot misfit
            data = self.datao1.flatten()
            data[self.index_data2] = (
                self.data_mod[self.n_data1 :] - self.data_ori[self.n_data1 :]
            )
            data -= np.nanmedian(data)
            data = data.reshape(self.data2_shape)
            vmin = np.ceil(np.nanquantile(data, 0.005) * 1000) / 1000
            vmax = np.floor(np.nanquantile(data, 0.995) * 1000) / 1000
            br_map, norm = utils.mag_color_map(vmin, vmax)
            im, cbar = utils.data_plot(
                data,
                self.fig_theo2.fig,
                self.ax_theo2[1],
                title="Mag difference sensor2\n"
                + f"rel. RMS misfit: {self.std_data_rel*100:0.2f}%",
                xtitle=f"Easting [{self.ax_unit}]",
                ytitle=f"Northing [{self.ax_unit}]",
                extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
                cbar_title="Difference calc-meas [nT]",
                cmap=br_map,
                norm=norm,
            )
            self.ax_theo2[1].grid(visible=True, which="both")
            self.fig_theo2.show()
            self.fig_theo2.fig.savefig(plt_file2)

        # Plot misfit evolution
        if self.iteration > 0:
            self.fig_misfit = newWindow("Magnetic data", 800, 500)
            self.ax_misfit = self.fig_misfit.fig.subplots(1, 1)
            self.ax_misfit.plot(
                np.arange(len(self.rel_RMS_misfit)), self.rel_RMS_misfit, "k"
            )
            self.ax_misfit.set_title("Misfit evolution")
            self.ax_misfit.set_xlabel("Iteration number")
            self.ax_misfit.set_ylabel("RMS misfit [%]")
            self.fig_misfit.show()
            self.fig_misfit.fig.savefig(
                os.path.join(self.folder, "misfit-evolution.png")
            )
        print(
            "\nClick into parameter distribution window and press ENTER "
            + "to finish inversion, r to continue iterations"
        )
        while True:
            event = self.fig_par.get_event()
            if event.name == "key_press_event":
                if event.key in ("enter", "r", "R"):
                    break
            # else:
            #     self.plot_cross_section(event.x, event.y)
        self.fig_par.close_window()
        self.fig_theo.close_window()
        if self.iteration > 0:
            self.fig_misfit.close_window()
        if self.n_sensor == 2:
            self.fig_theo2.close_window()
        if event.key == "enter":
            return False
        else:
            return True

    def plot_2D(self, file, inv_flag=False, coord="X"):
        cmap = plt.get_cmap("rainbow")
        if self.sus_inv:
            txt = "Susceptibility"
            title = "Magnetic data"
            ylabel = "Anomaly [nT]"
            if self.log_plot_flag:
                c_txt = "log10(Sus [SI]*10**6)"
            else:
                c_txt = "Sus [SI]*10**3"
        elif self.rem_inv:
            txt = "Remanence"
            title = "Magnetic data: "
            ylabel = "Anomaly [nT]"
            c_txt = "Remanence [A/m]"
        else:
            txt = "Density"
            title = "Gravity data"
            ylabel = "Anomaly [mGal]"
            c_txt = "Density [kg/m3]"
        if inv_flag:
            if self.direction in ("N", "S", "Y", 0.0, 180.0):
                title += f" at X={self.line_pos*self.dfac:0.2f}"
            else:
                title += f" at Y={self.line_pos*self.dfac:0.2f}"
            if self.iteration > 0:
                title += f"; RMS misfit: {self.std_data_rel*100:0.2f}%"
            if file[-1] == "N":
                sens = "Easting"
            else:
                sens = "Northing"
        else:
            sens = "Distance"
        self.fig_2D = newWindow(f"{txt} model", 2000, 1500, 15, 15)
        self.fig_2D.fig.tight_layout(w_pad=15, h_pad=2)
        self.gs = GridSpec(16, 10, self.fig_2D.fig)
        ax_dat = self.fig_2D.fig.add_subplot(self.gs[1:7, 1:9])
        ax_mod = self.fig_2D.fig.add_subplot(self.gs[10:16, 1:9])
        if coord == "X":
            xmin = self.x.min()
            xmax = self.x.max()
            x = np.copy(self.x)
        else:
            xmin = self.y.min()
            xmax = self.y.max()
            x = np.copy(self.y)
        if xmax - xmin > 10000:
            xx = x / 1000.0
            xmin /= 1000.0
            xmax /= 1000.0
            unit = "km"
            fac = 0.001
        else:
            xx = x
            unit = "m"
            fac = 1.0
        if self.n_sensor == 1:
            if inv_flag:
                dat = self.data_ori + np.nanmedian(self.data_mod)
                ax_dat.plot(xx, dat, "b*", label="meas. data")
            ax_dat.plot(xx, self.data_mod, "cyan", label="calc. data")
            if inv_flag:
                ax_dat.legend(bbox_to_anchor=(1, 1), loc="upper right", fontsize=10)
        else:
            if inv_flag:
                dat = (
                    self.data_ori[: self.n_data1]
                    + np.nanmedian(self.data_mod[: self.n_data1])
                    - np.nanmedian(self.data_ori[: self.n_data1])
                )
                ax_dat.plot(xx[: self.n_data1], dat, "b*", label="meas. data sensor 1")
                dat = (
                    self.data_ori[self.n_data1 :]
                    + np.nanmedian(self.data_mod[self.n_data1 :])
                    - np.nanmedian(self.data_ori[self.n_data1 :])
                )
                ax_dat.plot(xx[self.n_data1 :], dat, "r+", label="meas. data sensor 2")
            ax_dat.plot(
                xx[: self.n_data1],
                self.data_mod[: self.n_data1],
                "cyan",
                label="calc. data sensor 1",
            )
            ax_dat.plot(
                xx[self.n_data1 :],
                self.data_mod[self.n_data1 :],
                "orange",
                label="calc. data sensor 2",
            )
            if inv_flag:
                ax_dat.legend(bbox_to_anchor=(1, 1), loc="upper right", fontsize=10)
        ax_dat.tick_params(axis="both", labelsize=12)
        ax_dat.set_title(title, fontsize=14)
        ax_dat.set_xlabel(f"{sens} [{unit}]", fontsize=12)
        ax_dat.set_ylabel(ylabel, fontsize=12)
        ax_dat.set_xlim([xmin, xmax])
        # Plot model
        cax = ax_dat.inset_axes([1.05, 0.05, 0.02, 0.9], transform=ax_dat.transAxes)
        cax.axis("off")
        patches = []
        col = []
        ymin = 1e10
        ymax = -1e10
        if self.sus_inv:
            susmin = 1.0e10
            for _, val in self.mPrism.prisms.items():
                susmin = min(susmin, val.getsus())
        with open("prism_test.dat", "w") as fo:
            for key, val in self.mPrism.prisms.items():
                if self.sus_inv:
                    sus = val.getsus()
                    if self.log_plot_flag:
                        if susmin < 0.0:
                            sus -= susmin
                        col.append(np.log10(sus * 10**6 + 1.0))
                    else:
                        col.append(sus * 10**3)
                elif self.rem_inv:
                    col.append(val.rem)
                else:
                    col.append(val.rho)
                if val.typ == "O":
                    if coord == "X":
                        arr = [[val.x[0] * fac, (val.z[0] + val.z[1]) * fac / 2.0]]
                        arr.append([val.x[1] * fac, (val.z[2] + val.z[3]) * fac / 2.0])
                        arr.append([val.x[1] * fac, (val.z[6] + val.z[7]) * fac / 2.0])
                        arr.append([val.x[0] * fac, (val.z[4] + val.z[5]) * fac / 2.0])
                    else:
                        arr = [[val.y[0] * fac, (val.z[0] + val.z[1]) * fac / 2.0]]
                        arr.append([val.y[1] * fac, (val.z[2] + val.z[3]) * fac / 2.0])
                        arr.append([val.y[1] * fac, (val.z[6] + val.z[7]) * fac / 2.0])
                        arr.append([val.y[0] * fac, (val.z[4] + val.z[5]) * fac / 2.0])
                    arr.append(arr[0])
                    arr = np.array(arr)
                    fo.write(f"{key}: {arr}, sus: {val.getsus()}\n")
                    patches.append(Polygon(arr))
                    ymin = min(ymin, arr[:, 1].min())
                    ymax = max(ymax, arr[:, 1].max())
                else:
                    if coord == "X":
                        x1 = val.x[0] * fac
                        x2 = val.x[1] * fac
                    else:
                        x1 = val.y[0] * fac
                        x2 = val.y[1] * fac
                    y1 = val.z[0] * fac
                    y2 = val.z[1] * fac
                    ymin = min(ymin, y1)
                    ymax = max(ymax, y2)
                    patches.append(Rectangle((x1, y1), x2 - x1, y2 - y1))
        col = np.array(col)
        p = PatchCollection(patches, cmap=cmap)
        p.set_array(col)
        ax_mod.add_collection(p)
        if self.topo_flag:
            ax_mod.plot(self.x * fac, self.z * fac, "r")
            ymin = min(ymin, self.z.min() * fac)
        for key, val in self.mPrism.prisms.items():
            if val.typ == "O":
                xp = np.array([val.x[0], val.x[1], val.x[1], val.x[0], val.x[0]])
                yp = np.array([val.z[0], val.z[2], val.z[6], val.z[4], val.z[0]])
            else:
                xp = np.array([val.x[0], val.x[1], val.x[1], val.x[0], val.x[0]])
                yp = np.array([val.z[0], val.z[0], val.z[1], val.z[1], val.z[0]])
            xp *= fac
            yp *= fac
            ax_mod.plot(xp, yp, "k")
        ax_mod.set_xlim([xmin, xmax])
        ax_mod.set_ylim([ymin, ymax])
        ax_mod.invert_yaxis()
        ax_mod.tick_params(axis="both", labelsize=12)
        ax_mod.set_title(txt, fontsize=14)
        ax_mod.set_xlabel(f"{sens} [{unit}]", fontsize=12)
        ax_mod.set_ylabel(f"Depth [{unit}]", fontsize=12)
        if self.equal_flag:
            ax_mod.set_aspect("equal")
        else:
            ax_mod.set_aspect("auto")
        cax = ax_mod.inset_axes([1.05, 0.05, 0.02, 0.9], transform=ax_mod.transAxes)
        # Plot color bar
        cbar = plt.colorbar(p, orientation="vertical", cax=cax, fraction=0.1)
        cbar.set_label(label=c_txt, size=12)
        for lab in cbar.ax.yaxis.get_ticklabels():
            lab.set_fontsize(12)
        self.fig_2D.setHelp(
            "Press ENTER to finish; press r to continue iterations; "
            + "press e to toggle between equal scale and filling window "
            + "for model axis; press c to toggle between linear and log "
            + "color scale. If no reaction, move mouse a little bit"
        )
        self.fig_2D.show()

    def show_synthetic(self):
        """
        Plots results from synthetic model calculation

        """
        self.equal_flag = False
        if "m" in self.data_type:
            self.fig_syn = newWindow("Magnetic data", 1500, 1000, 20, 15)
            self.fig_syn.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Mag_synthetic.png")
            title = f"Modelled magnetic anomaly {self.z[0]}m"
            if self.n_sensor == 2:
                title2 = f"Modelled magnetic anomaly {self.z2[0]}m"
            unit = "nT"
        else:
            self.fig_syn = newWindow("Gravity data", 1500, 1000, 20, 15)
            self.fig_syn.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Gravi_synthetic.png")
            title = "Modelled gravity anomaly"
            unit = "mGal"
        # Plot magnetic anomalies produced by synthetic model
        data = self.data_mod[: self.n_data1].reshape(self.data1_shape)
        if self.data1_shape[0] == 1 or self.data_shape[1] == 1:
            if self.data1_shape[0] == 1:
                self.direction = "E"
                coord = "X"
            else:
                self.direction = "N"
                coord = "Y"
            self.plot_2D(plt_file, coord=coord)
            print("\nClick to close window and finish synthetic calculation")
            while True:
                event = self.fig_2D.get_event()
                if event.name == "button_press_event":
                    self.fig_2D.close_window()
                    break
        else:
            if self.n_sensor == 2:
                data2 = self.data_mod[self.n_data1 :].reshape(self.data2_shape)
                ddx = self.x.max() - self.x.min()
                ddy = self.y.max() - self.y.min()
                facx = 10 / (2 * ddx)
                facy = 8 / (2 * ddy)
                # Vertical layout
                self.ax_syn = []
                if facx < facy:
                    self.gs = GridSpec(18, 10, self.fig_syn.fig)
                    self.ax_syn.append(self.fig_syn.fig.add_subplot(self.gs[1:8, 1:]))
                    self.ax_syn.append(self.fig_syn.fig.add_subplot(self.gs[10:17, 1:]))
                    self.bar_or = "vertical"
                    self.nticks = 10
                # Horizontal layout
                else:
                    self.gs = GridSpec(10, 18, self.fig_syn.fig)
                    self.ax_syn.append(self.fig_syn.fig.add_subplot(self.gs[1:, 1:8]))
                    self.ax_syn.append(self.fig_syn.fig.add_subplot(self.gs[1:, 10:17]))
                    self.bar_or = "horizontal"
                    self.nticks = 5
            else:
                self.gs = GridSpec(10, 10, self.fig_syn.fig)
                self.ax_syn = [self.fig_syn.fig.add_subplot(self.gs[1:-1, 1:-1])]
            vmin = np.ceil(np.nanquantile(data, 0.005) * 1000) / 1000
            vmax = np.ceil(np.nanquantile(data, 0.995) * 1000) / 1000
            self.xplt_min = self.x.min()
            self.xplt_max = self.x.max()
            self.yplt_min = self.y.min()
            self.yplt_max = self.y.max()
            if "m" in self.data_type:
                br_map, norm = utils.mag_color_map(vmin, vmax)
                im, cbar = utils.data_plot(
                    data,
                    self.fig_syn.fig,
                    self.ax_syn[0],
                    title=f"{title}",
                    xtitle="Easting [m]",
                    ytitle="Northing [m]",
                    cmap=br_map,
                    norm=norm,
                    cbar_title=unit,
                    extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
                )
                if self.n_sensor == 2:
                    im, cbar = utils.data_plot(
                        data2,
                        self.fig_syn.fig,
                        self.ax_syn[1],
                        title=f"{title2}",
                        xtitle="Easting [m]",
                        ytitle="Northing [m]",
                        cmap=br_map,
                        norm=norm,
                        cbar_title=unit,
                        extent=[
                            self.xplt_min,
                            self.xplt_max,
                            self.yplt_min,
                            self.yplt_max,
                        ],
                    )
            else:
                im, cbar = utils.data_plot(
                    data,
                    self.fig_syn.fig,
                    self.ax_syn[0],
                    title=f"{title}",
                    xtitle="Easting [m]",
                    ytitle="Northing [m]",
                    cmap="rainbow",
                    cbar_title=unit,
                    extent=[self.xplt_min, self.xplt_max, self.yplt_min, self.yplt_max],
                )
            # plot prism contours
            for key, val in self.mPrism.prisms.items():
                x1 = val.x[0]
                x2 = val.x[1]
                y1 = val.y[0]
                y2 = val.y[1]
                self.ax_syn[0].plot(
                    [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k", linewidth=1
                )
                if self.n_sensor == 2:
                    self.ax_syn[1].plot(
                        [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k", linewidth=1
                    )
            self.ax_syn[0].set_xlim([self.xplt_min, self.xplt_max])
            self.ax_syn[0].set_ylim([self.yplt_min, self.yplt_max])
            self.ax_syn[0].grid(visible=True, which="both")
            if self.n_sensor == 2:
                self.ax_syn[1].set_xlim([self.xplt_min, self.xplt_max])
                self.ax_syn[1].set_ylim([self.yplt_min, self.yplt_max])
                self.ax_syn[1].grid(visible=True, which="both")
            self.fig_syn.fig.savefig(plt_file)
            self.fig_syn.show()
            print("\nClick to close window and finish synthetic calculation")
            while True:
                event = self.fig_syn.get_event()
                if event.name == "button_press_event":
                    break
            self.fig_syn.close_window()
        with open(os.path.join(self.folder, "synthetic_data.dat"), "w") as fo:
            ny, nx = self.data1_shape
            nd = nx * ny
            x = self.x[:nd].reshape(self.data1_shape)
            y = self.y[:nd].reshape(self.data1_shape)
            z = -self.z[:nd].reshape(self.data1_shape)
            fo.write("Synthetic data\n")
            fo.write(f"{nx} {ny}\n")
            if self.n_sensor == 1:
                for ix in range(nx):
                    for iy in range(ny):
                        fo.write(
                            f"{x[iy, ix]:0.3f} {y[iy, ix]:0.3f} "
                            + f"{z[iy, ix]:0.3f} {data[iy, ix]:0.3f}\n"
                        )
            else:
                for ix in range(nx):
                    for iy in range(ny):
                        fo.write(
                            f"{x[iy, ix]:0.3f} {y[iy, ix]:0.3f} "
                            + f"{z[iy, ix]:0.3f} {data[iy, ix]:0.3f} "
                            + f"{data2[iy, ix]:0.3f}\n"
                        )

    def save_model(self):
        """
        Writes model parameters into two files:

        model.dat contains prism centers and properties to draw a map

        synthetic_model.txt contains full prism coordinates and properties,
        may be modified manually and used for testing of synthetic models.

        Returns
        ----------
        None
        """
        with open(os.path.join(self.folder, "model.dat"), "w") as fo:
            text1 = ""
            text2 = ""
            if self.sus_inv:
                text1 += "     sus    "
                text2 += "  [1E-6 SI] "
            if self.rem_inv:
                text1 += "  rem  "
                text2 += " [A/m] "
            if self.rho_inv:
                text1 += "  rho  "
                text2 += "[km/m3]"
            fo.write(
                f"           Prism_center      {text1}                   "
                + " Prism coordinates\n"
            )
            if self.direction == "N":
                fo.write(
                    f"      X          Y       Z   {text2}   #       S   "
                    + "      N          W          E      top   bottom\n"
                )
            else:
                fo.write(
                    f"      X          Y       Z   {text2}   #       W   "
                    + "      E          S          N      top   bottom\n"
                )
            for key, val in self.mPrism.prisms.items():
                if self.direction in ("N", "S", 0.0, 180.0) and self.dim == 2:
                    x1 = val.y[0]
                    x2 = val.y[1]
                    xc = (x1 + x2) * 0.5
                    y1 = val.x[0]
                    y2 = val.x[1]
                    yc = (y1 + y2) * 0.5
                else:
                    x1 = val.x[0]
                    x2 = val.x[1]
                    xc = (x1 + x2) * 0.5
                    y1 = val.y[0]
                    y2 = val.y[1]
                    yc = (y1 + y2) * 0.5
                if val.typ == "P":
                    z1 = val.z[0]
                    z2 = val.z[1]
                else:
                    z1 = np.mean(val.z[:4])
                    z2 = np.mean(val.z[4:])
                zc = (z1 + z2) * 0.5
                fo.write(f"{xc:9.2f} {yc:10.2f} {zc:7.2f} ")
                if self.sus_inv:
                    fo.write(f"{val.getsus()*1E6:10.0f}")
                if self.rem_inv:
                    fo.write(f"{val.rem:7.3f}")
                if self.rho_inv:
                    fo.write(f"{val.rho:7.0f}")
                if self.direction in ("N", "S", 0.0, 180.0) and self.dim == 2:
                    fo.write(
                        f"{key:6d} {y1:9.2f} {y2:9.2f} {x1:10.2f} "
                        + f"{x2:10.2f} {z1:7.2f} {z2:7.2f}\n"
                    )
                else:
                    fo.write(
                        f"{key:6d} {x1:9.2f} {x2:9.2f} {y1:10.2f} "
                        + f"{y2:10.2f} {z1:7.2f} {z2:7.2f}\n"
                    )
        # Write model in format to be used as synthetic model. Allows modifying it
        # manually for tests
        with open(os.path.join(self.folder, "synthetic_model.txt"), "w") as fo:
            for key, val in self.mPrism.prisms.items():
                if val.typ == "P":
                    z1 = val.z[0]
                    z2 = val.z[1]
                else:
                    z1 = np.mean(val.z[:4])
                    z2 = np.mean(val.z[4:])
                if self.direction in ("N", "S", 0.0, 180.0) and self.dim == 2:
                    fo.write(
                        f"{val.y[0]:0.1f} {val.y[1]:0.1f} {val.x[0]:0.1f}"
                        + f" {val.x[1]:0.1f} {z1:0.1f} {z2:0.1f} "
                        + f"{val.getsus():0.6f} "
                        + f"{val.rem:0.3f} {val.inc:0.1f} "
                        + f"{val.dec:0.1f} {val.rho:0.1f}\n"
                    )
                else:
                    fo.write(
                        f"{val.x[0]:0.1f} {val.x[1]:0.1f} {val.y[0]:0.1f}"
                        + f" {val.y[1]:0.1f} {z1:0.1f} "
                        + f"{z2:0.1f} {val.getsus():0.6f} "
                        + f"{val.rem:0.3f} {val.inc:0.1f} "
                        + f"{val.dec:0.1f} {val.rho:0.1f}\n"
                    )
        return None

    def get_inversion_parameters(self, data_type):
        """
        Ask for inversion control parameters

        Parameters
        ----------
        data_type : str
            Data to be inverted: "m1" or "m2" for magnetic data,
            "g" for gravity data

        Defines the following variables
        -------------------------------
        sus_inv, rem_inv, rho_inv : booleans
            If True, susceptibility, remanence or density inversion activated
        max_iter : int
            Maximum number of iterations
        max_diff_fac : float
            Iterations stop if maximum relative RMS fit of each data set is
            smaller than the given value [%/100]
        max_rel_diff : float
            Iterations stop if variation of maximum relative RMS fit of each
            data set from one iteration to the next is smaller than the given
            value [%/100]
        lam, lam_fac, lam_min: floats
            initial lambda, factor per iteration step, smallest allowed lambda
        gam, gam_fac, gam_min: floats
            initial gamma, factor per iteration step, smallest allowed gamma

        """
        results = False
        if self.iteration < 0:
            itera = self.add_iter
            pos = None
        else:
            itera = self.iteration + self.add_iter
            if self.positive:
                pos = 1
            else:
                pos = None
        if "m" in data_type:
            results, okButton = dialog(
                [
                    "Invert for:",
                    "Susceptibility",
                    "Remanence",
                    "Maximum number of iterations",
                    "Maximum relative RMS misfit",
                    "Maximum variation of RMS",
                    "Initial lambda (regularization)",
                    "Lambda factor per iteration",
                    "Minimum lambda",
                    "Initial Gamma (smoothing)",
                    "Gamma factor per iteration",
                    "Minimum Gamma",
                    "Use positivity constraint",
                ],
                ["l", "c", "c", "e", "e", "e", "e", "e", "e", "e", "e", "e", "c"],
                [
                    None,
                    1,
                    0,
                    itera,
                    self.max_diff_fac,
                    self.max_rel_diff,
                    self.lam,
                    self.lam_fac,
                    self.lam_min,
                    self.gam,
                    self.gam_fac,
                    self.gam_min,
                    pos,
                ],
                "Magnetic inversion parameters",
            )
        else:
            results, okButton = dialog(
                [
                    "Maximum number of iterations",
                    "Maximum relative RMS misfit",
                    "Maximum variation of RMS",
                    "Initial lambda (regularization)",
                    "Lambda factor per iteration",
                    "Minimum lambda",
                    "Initial Gamma (smoothing)",
                    "Gamma factor per iteration",
                    "Minimum Gamma",
                    "Use positivity constraint",
                ],
                ["e", "e", "e", "e", "e", "e", "e", "e", "e", "c"],
                [
                    itera,
                    self.max_diff_fac,
                    self.max_rel_diff,
                    self.lam,
                    self.lam_fac,
                    self.lam_min,
                    self.gam,
                    self.gam_fac,
                    self.gam_min,
                    pos,
                ],
                "Gravity inversion parameters",
            )
        if not okButton:
            print("No inversion parameters given")
            return False
        # set flags for properties to be optimized
        if "m" in data_type:
            if int(results[1]) > -1:
                self.sus_inv = True
            if int(results[2]) > -1:
                self.rem_inv = True
            if self.sus_inv and self.rem_inv:
                print(
                    "\nInverting for both, susceptibility and remanence, "
                    + "does not make sense.\n\nSusceptibility inversion "
                    + "canceled and susceptibility set to zero."
                )
                self.sus_inv = False
            if not self.sus_inv and not self.rem_inv:
                print(
                    "\nNo parameter type for inversion chosen.\n"
                    + "Susceptibility will be used by default"
                )
                self.sus_inv = True
            ianswer = 3
        else:
            self.rho_inv = True
            ianswer = 0
        # Set maximum number of iterations
        self.max_iter = int(results[ianswer])
        ianswer += 1
        # set relative RMS misfit for all data sets which stops iterations if reached
        self.max_diff_fac = float(results[ianswer])
        ianswer += 1
        # set maximum relative change of RMS misfit from one iteration to the next for
        #     which iterations continue
        self.max_rel_diff = float(results[ianswer])
        ianswer += 1
        # set relative importance of initial model and reduction of this parameter per
        # iteration
        self.lam = float(results[ianswer])
        ianswer += 1
        self.lam_fac = float(results[ianswer])
        ianswer += 1
        self.lam_min = float(results[ianswer])
        ianswer += 1
        self.gam = float(results[ianswer])
        ianswer += 1
        self.gam_fac = float(results[ianswer])
        ianswer += 1
        self.gam_min = float(results[ianswer])
        ianswer += 1
        if int(results[ianswer]) > -1:
            self.positive = True
        else:
            self.positive = False
        return True

    def get_area2D(self):
        """
        Define model space and initial prism sizes for 2.5D model.
        If block sizes are reduced in certain areas, minimum accepted block
        sizes are stored in min_size_x, .._y, .._z

        Defines the following variables
        -------------------------------
        xprism_min: float
            Smallest x-coordinate for prism limits
        xprism_max: float
            Largest x-coordinate for prism limits
        dx_prism: float
            Initial prism size in x
        min_size_x: float
            Minimum allowed prism size in x

        Similar variable for the y and z direction
        """
        #    Calculate default prism sizes
        xmin = self.x.min()
        xmax = self.x.max()
        line_length = xmax - xmin
        dx = line_length / (len(self.x) - 1)
        nr = 1 - int(np.log10(line_length))
        while True:
            dx_ini = np.round(line_length / 5.0, nr)
            if dx_ini > 0.0:
                break
            nr += 1
        dz_ini = dx_ini / 2.0
        zmin = 0
        zmax = dz_ini
        yw = zmax

        nx = int(np.ceil(line_length / dx_ini))
        if dx_ini * nx - dx <= line_length:
            nx += 1

        while True:
            results, okButton = dialog(
                [
                    "xmin [m]",
                    "xmax [m]",
                    "dx_ini [m]",
                    "minimum size x [m]",
                    "zmin [m]",
                    "zmax [m]",
                    "dz_ini [m]",
                    "minimum size z [m]",
                    "prism half-width [m]",
                    "Data point reduction\n(1 point out of...)",
                    "Use topography",
                ],
                ["e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "c"],
                [
                    xmin,
                    xmax,
                    dx_ini,
                    dx_ini / 4.0,
                    zmin,
                    zmax,
                    dz_ini,
                    dz_ini / 4.0,
                    yw,
                    1,
                    0,
                ],
                "2D prism parameters",
            )
            if not okButton:
                print("\nNo prism data given, inversion aborted")
                return False
            self.xdata_min = float(results[0])
            self.xdata_max = float(results[1])
            self.dx_prism = float(results[2])
            self.min_size_x = float(results[3])
            self.zprism_min = float(results[4])
            self.zprism_max = float(results[5])
            self.dz_prism = float(results[6])
            self.min_size_z = float(results[7])
            self.data_reduction = int(results[9])
            self.topo_flag = int(results[10]) > -1
            yw = float(results[8])
            self.yprism_min = -yw
            self.yprism_max = yw
            self.dy_prism = 2.0 * yw
            self.min_size_y = self.dy_prism
            # To avoid boundary effects prisms are defined on an area larger than the data
            # area by at least the maximum model depth.
            self.xprism_min = self.xdata_min - self.zprism_max
            self.xprism_max = self.xdata_max + self.zprism_max
            # Set limits of prism zone to multiple of initial prism size
            xlen0 = self.xprism_max - self.xprism_min
            nxp = int(xlen0 / self.dx_prism)
            if nxp * self.dx_prism < xlen0:
                nxp += 1
                xlen = nxp * self.dx_prism
                dx = (xlen - xlen0) * 0.5
                self.xprism_min -= dx
                self.xprism_max += dx
            ret = self.prepare_data()
            if ret is True:
                break
        self.set_prisms()
        # self.set_prisms_test()
        return True

    def get_area3D(self):
        """
        Define model space and initial prism sizes for 3D model.
        If block sizes are reduced in certain areas, minimum accepted block
        sizes are stored in min_size_x, .._y, .._z

        Defines the following variables
        -------------------------------
        xprism_min: float
            Smallest x-coordinate for prism limits
        xprism_max: float
            Largest x-coordinate for prism limits
        dx_prism: float
            Initial prism size in x
        min_size_x: float
            Minimum allowed prism size in x

        Similar variable for the y and z direction

        """
        #    Calculate default prism sizes
        xmin = self.x.min()
        xmax = self.x.max()
        line_length_x = xmax - xmin
        dx = line_length_x / (len(self.x) - 1)
        ymin = self.y.min()
        ymax = self.y.max()
        line_length_y = ymax - ymin
        dy = line_length_y / (len(self.y) - 1)
        length_xy = max((line_length_x, line_length_y))
        nr = 1 - int(np.log10(length_xy))
        dx_ini = np.round(np.sqrt(line_length_x * line_length_y / 25.0), nr)
        dy_ini = dx_ini
        dz_ini = length_xy / 10.0
        zmin = 0.0
        zmax = dz_ini
        nx = int(np.ceil(line_length_x / dx_ini))
        if dx_ini * nx - dx <= line_length_x:
            nx += 1

        ny = int(np.ceil(dy / dy_ini))
        if dy_ini * ny - dy <= line_length_y:
            ny += 1

        # Get initial prism configuration and prism control parameters
        while True:
            results, okButton = dialog(
                [
                    "xmin [m]",
                    "xmax [m]",
                    "dx_ini [m]",
                    "minimum size x [m]",
                    "ymin [m]",
                    "ymax [m]",
                    "dy_ini [m]",
                    "minimum size y [m]",
                    "zmin [m]",
                    "zmax [m]",
                    "dz_ini [m]",
                    "minimum size z [m]",
                    "Data point reduction\n(1 point out of...)",
                    "Use topography",
                ],
                ["e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "c"],
                [
                    xmin,
                    xmax,
                    dx_ini,
                    dx_ini / 4,
                    ymin,
                    ymax,
                    dy_ini,
                    dy_ini / 4,
                    zmin,
                    zmax,
                    dz_ini,
                    dz_ini / 4,
                    1,
                    0,
                ],
                "Prism parameters",
            )
            if not okButton:
                print("\nNo prism data given, inversion aborted")
                return False
            self.xdata_min = float(results[0])
            self.xdata_max = float(results[1])
            self.dx_prism = float(results[2])
            self.min_size_x = float(results[3])
            self.ydata_min = float(results[4])
            self.ydata_max = float(results[5])
            self.dy_prism = float(results[6])
            self.min_size_y = float(results[7])
            self.zprism_min = float(results[8])
            self.zprism_max = float(results[9])
            self.dz_prism = float(results[10])
            self.min_size_z = float(results[11])
            self.data_reduction = int(results[12])
            self.topo_flag = int(results[13]) > -1
            # To avoid boundary effects prisms are defined on an area larger than the data
            # area by at least the maximum model depth.
            self.xprism_min = self.xdata_min - self.zprism_max
            self.xprism_max = self.xdata_max + self.zprism_max
            # Set limits of prism zone to multiple of initial prism size
            xlen0 = self.xprism_max - self.xprism_min
            nxp = int(xlen0 / self.dx_prism)
            if nxp * self.dx_prism < xlen0:
                nxp += 1
                xlen = nxp * self.dx_prism
                dx = (xlen - xlen0) * 0.5
                self.xprism_min -= dx
                self.xprism_max += dx
            self.yprism_min = self.ydata_min - self.zprism_max
            self.yprism_max = self.ydata_max + self.zprism_max
            ylen0 = self.yprism_max - self.yprism_min
            nyp = int(ylen0 / self.dy_prism)
            if nyp * self.dy_prism < ylen0:
                nyp += 1
                ylen = nyp * self.dy_prism
                dy = (ylen - ylen0) * 0.5
                self.yprism_min -= dy
                self.yprism_max += dy
            # Calculate effective minimum prism thickness and round to the nearest smaller
            # meter or, if thinner prisms are desired, to the nearest smaller decimeter.
            dz = self.zprism_max - self.zprism_min
            while True:
                dz /= 2.0
                if dz < self.min_size_z:
                    break
            if self.min_size_z >= 1.0:
                self.min_size_z = np.floor(dz * 2.0)
            else:
                self.min_size_z = np.floor(dz * 2.0 * 10.0) / 10.0
            dx = self.dx_prism
            while True:
                dx *= 0.5
                if dx < self.min_size_x:
                    dx *= 2.0
                    break
            dx_min = dx
            dy = self.dy_prism
            while True:
                dy *= 0.5
                if dy < self.min_size_y:
                    dy *= 2.0
                    break
            dy_min = dy
            # make sure there are topography data over the whole prism area
            self.xtopo = np.arange(
                self.xprism_min, self.xprism_max + dx_min / 2, dx_min
            )
            self.ytopo = np.arange(
                self.yprism_min, self.yprism_max + dy_min / 2, dy_min
            )
            self.nx_topo = len(self.xtopo)
            self.ny_topo = len(self.ytopo)
            X, Y = np.meshgrid(self.xtopo, self.ytopo)
            if self.topo_flag:
                self.topo = self.topo_inter(X.flatten(), Y.flatten())
                t = self.topo.reshape(self.ny_topo, self.nx_topo)
                if np.isnan(t.max()):
                    for i in range(self.ny_topo):
                        if np.isnan(np.max(t[i, 0])) and np.isfinite(t[i, -1]):
                            continue
                        for j, tt in enumerate(t[i, :]):
                            if np.isfinite(tt):
                                break
                        if j > 0:
                            t[i, :j] = t[i, j]
                        for j, tt in enumerate(t[i, ::-1]):
                            if np.isfinite(tt):
                                break
                        if j > 0:
                            t[i, -j:] = t[i, -(j + 1)]
                    for j in range(self.nx_topo):
                        if np.isfinite(t[0, j]) and np.isfinite(t[-1, j]):
                            continue
                        for i, tt in enumerate(t[:, j]):
                            if np.isfinite(tt):
                                break
                        if i > 0:
                            t[:i, j] = t[i, j]
                        for i, tt in enumerate(t[::-1, j]):
                            if np.isfinite(tt):
                                break
                        if i > 0:
                            t[-i:, j] = t[-(i + 1), j]
                self.topo = t
                self.topo_inter = interpolate.NearestNDInterpolator(
                    list(zip(X.flatten(), Y.flatten())), self.topo.flatten()
                )
            # Extract data within prism area and, if asked for, reduce point density
            ret = self.prepare_data()
            if ret is True:
                break
        # Back up original (reduced) data
        self.data_ori = np.copy(self.data)
        self.std_data_ori = np.std(self.data)
        self.xmin = np.nanmin((self.x))
        self.xmax = np.nanmax((self.x))
        self.ymin = np.nanmin((self.y))
        self.ymax = np.nanmax((self.y))
        self.dx = (xmax - xmin) / (self.data_shape[1] - 1)
        self.dy = (ymax - ymin) / (self.data_shape[0] - 1)
        self.set_prisms()
        return True

    def prepare_data(self):
        """
        Extract data used for inversion from full data set and concatenate
        all data into one flattened numpy 1D array.
        Data reduction may be due to restriction of coordinates and/or due to
        reduction of number of data (taking only one point out of several in
        both directions).

        Reduction is done in situ, i.e. self.x, self.y, self.z, self.data and
        self.data_shape are modified.
        """
        if self.dim == 2 or len(self.data_shape) == 1:
            ndata = self.data1_shape[0]
            xdata_min = self.xdata_min
            xdata_max = self.xdata_max
            n = np.where(self.x[:ndata] >= xdata_min)[0]
            if len(n) > 0:
                n1 = n[0]
            else:
                _ = QtWidgets.QMessageBox.warning(
                    None,
                    "Warning",
                    "No data available for prism area:\n"
                    + f"xmin_prism = {xdata_min:0.3f}\n"
                    + f"xmax_data = {self.x[:ndata].max():0.3f}\n\n"
                    + "Redefine area",
                    QtWidgets.QMessageBox.Close,
                    QtWidgets.QMessageBox.Close,
                )
                return False
            n = np.where(self.x[:ndata] <= xdata_max)[0]
            if len(n) > 0:
                n2 = n[-1] + 1
            else:
                _ = QtWidgets.QMessageBox.warning(
                    None,
                    "Warning",
                    "No data available for prism area:\n"
                    + f"xmax_prism = {xdata_max:0.3f}\n"
                    + f"xmin_data = {self.x[:ndata].min():0.3f}\n\n"
                    + "Redefine area",
                    QtWidgets.QMessageBox.Close,
                    QtWidgets.QMessageBox.Close,
                )
                return False
            for i in range(n1, n2):
                if np.isfinite(self.data[i]):
                    break
            n1 = i
            for i in range(n2 - 1, n1 - 1, -1):
                if np.isfinite(self.data[i]):
                    break
            n2 = i + 1
            n1 = max(n1, int(self.data_reduction / 2))
            x = self.x[n1 : n2 : self.data_reduction]
            y = self.y[n1 : n2 : self.data_reduction]
            z = self.z[n1 : n2 : self.data_reduction]
            self.xo1 = np.copy(x)
            self.yo1 = np.copy(y)
            self.zo1 = np.copy(z)
            data = self.data[n1 : n2 : self.data_reduction]
            self.data_shape = data.shape
            self.data1_shape = data.shape
            data = data.flatten()
            self.index_data = np.where(np.isfinite(data))[0]
            self.datao1 = np.copy(data)
            data = data[self.index_data]
            x = x[self.index_data]
            y = y[self.index_data]
            z = z[self.index_data]
            self.n_data1 = len(data)
            if self.n_sensor == 2:
                n1 = np.where(self.x2[:ndata] >= xdata_min)[0][0]
                n2 = np.where(self.x2[:ndata] <= xdata_max)[0][-1] + 1
                for i in range(n1, n2):
                    if np.isfinite(self.data2[i]):
                        break
                n1 = i
                for i in range(n2 - 1, n1 - 1, -1):
                    if np.isfinite(self.data2[i]):
                        break
                n2 = i + 1
                n1 = max(n1, int(self.data_reduction / 2))
                d2 = self.data2[n1 : n2 : self.data_reduction]
                self.index_data2 = np.where(np.isfinite(d2))[0]
                self.datao2 = np.copy(d2)
                d2 = d2[self.index_data2]
                self.n_data2 = len(d2)
                self.data2_shape = d2.shape
                data = np.concatenate((data, d2))
                xx = self.x2[n1 : n2 : self.data_reduction]
                self.xo2 = np.copy(xx)
                x = np.concatenate((x, xx[self.index_data2]))
                yy = self.y2[n1 : n2 : self.data_reduction]
                self.yo2 = np.copy(yy)
                y = np.concatenate((y, yy[self.index_data2]))
                zz = self.z2[n1 : n2 : self.data_reduction]
                self.zo2 = np.copy(zz)
                z = np.concatenate((z, zz[self.index_data2]))
            else:
                self.n_data2 = 0
        else:
            ndata_y, ndata_x = self.data_shape
            ndata = ndata_x * ndata_y
            xdata_min = self.xdata_min
            xdata_max = self.xdata_max
            ydata_min = self.ydata_min
            ydata_max = self.ydata_max
            x = self.xo1.reshape(self.data_shape)
            y = self.yo1.reshape(self.data_shape)
            z = self.zo1.reshape(self.data_shape)
            d = self.datao1.reshape(self.data_shape)
            xcol = np.unique(self.xo1)
            yrow = np.unique(self.yo1)
            nx1 = np.where(xcol >= xdata_min)[0][0]
            nx2 = np.where(xcol <= xdata_max)[0][-1] + 1
            nx1 = max(nx1, int(self.data_reduction / 2))
            ny1 = np.where(yrow >= ydata_min)[0][0]
            ny2 = np.where(yrow <= ydata_max)[0][-1] + 1
            ny1 = max(ny1, int(self.data_reduction / 2))
            x = x[
                ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction
            ].flatten()
            y = y[
                ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction
            ].flatten()
            z = z[
                ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction
            ].flatten()
            data = d[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
            self.data_shape = data.shape
            self.data1_shape = data.shape
            data = data.flatten()
            self.index_data = np.where(np.isfinite(data))[0]
            self.datao1 = np.copy(data)
            data = data[self.index_data]
            self.xo1 = np.copy(x)
            self.yo1 = np.copy(y)
            self.zo1 = np.copy(z)
            x = x[self.index_data]
            y = y[self.index_data]
            z = z[self.index_data]
            self.n_data1 = len(data)
            if self.n_sensor == 2:
                d = np.copy(self.datao2)
                x2 = np.copy(self.xo2)
                y2 = np.copy(self.yo2)
                z2 = np.copy(self.zo2)
                xcol = np.unique(self.x2)
                yrow = np.unique(self.y2)
                nx1 = np.where(xcol >= xdata_min)[0][0]
                nx2 = np.where(xcol <= xdata_max)[0][-1] + 1
                nx1 = max(nx1, int(self.data_reduction / 2))
                ny1 = np.where(yrow >= ydata_min)[0][0]
                ny2 = np.where(yrow <= ydata_max)[0][-1] + 1
                ny1 = max(ny1, int(self.data_reduction / 2))
                self.xo2 = np.copy(
                    x2[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
                )
                self.yo2 = np.copy(
                    y2[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
                )
                self.zo2 = np.copy(
                    z2[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
                )
                self.datao2 = np.copy(
                    x2[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
                )
                self.data2_shape = self.datao2.shape
                self.index_data2 = np.where(np.isifinte(self.data2.flatten()))
                self.x2 = self.xo2.flatten()[self.index_data2]
                self.y2 = self.yo2.flatten()[self.index_data2]
                self.z2 = self.zo2.flatten()[self.index_data2]
                self.data2 = self.datao2.flatten()[self.index_data2]
                x = np.concatenate((x, self.x2))
                y = np.concatenate((y, self.y2))
                z = np.concatenate((z, self.z2))
                d2 = d[ny1 : ny2 : self.data_reduction, nx1 : nx2 : self.data_reduction]
                data = np.concatenate((data, self.data2))
                self.n_data2 = len(self.data2)
        self.x = x
        self.y = y
        self.z = z
        self.data = data
        print(f"\n{len(data)} data points to be inverted")
        self.data_ori = np.copy(data)
        self.data_ori -= np.nanmedian(self.data_ori)
        self.std_data_ori = np.std(self.data_ori)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xplt_min = self.x.min() - self.dx * 0.5
        self.xplt_max = self.x.max() + self.dx * 0.5
        self.yplt_min = self.y.min() - self.dy * 0.5
        self.yplt_max = self.y.max() + self.dy * 0.5
        if (
            max(self.xprism_max - self.xprism_min, self.yprism_max - self.yprism_min)
            > 10000
        ):
            self.xplt_min /= 1000.0
            self.xplt_max /= 1000.0
            self.yplt_min /= 1000.0
            self.yplt_max /= 1000.0
            self.ax_unit = "km"
        else:
            self.ax_unit = "m"
        return True

    def set_plot_depths(self):
        """
        Calculate depths at which horizontal planes through the final model
        shall be plotted

        Returns
        -------
        None.

        """
        # Define depths at which resulting model should be plotted (list of depths)
        if self.topo_flag:
            zm = self.z.max()
            self.z_plot = [zm - 1.0]
            self.z_plot += list(
                np.arange(
                    self.z.max() + self.min_size_z / 2,
                    self.zprism_max + zm,
                    self.min_size_z,
                )
            )
            self.z_plot = np.array(self.z_plot)
            self.nz_plot = []
            zmn = self.z_plot[0] - self.min_size_z / 2.0
            for i, zz in enumerate(self.z_plot):
                self.nz_plot.append(int((zz - zmn) / self.min_size_z))
            return
        self.z_plot = np.arange(
            self.zprism_min + self.min_size_z / 2, self.zprism_max, self.min_size_z
        )
        self.nz_plot = []
        for i, zz in enumerate(self.z_plot):
            self.nz_plot.append(int((zz - self.zprism_min) / self.min_size_z))

    def set_prisms(self):
        """
        Define prisms for initial model

        Returns
        -------
        None.

        """
        # Define initial model
        self.x_prism = np.arange(
            self.xprism_min, self.xprism_max + self.dx_prism / 2, self.dx_prism
        )
        zmin = np.round(self.topo.max(), 0)
        self.z_prism = (
            np.arange(
                self.zprism_min, self.zprism_max + self.dz_prism / 2, self.dz_prism
            )
            + zmin
        )
        self.mod_xshape = len(self.x_prism) - 1
        self.mod_zshape = len(self.z_prism) - 1
        if self.topo_flag:
            self.mod_zshape += 1
        if self.dim == 3:
            self.y_prism = np.arange(
                self.yprism_min, self.yprism_max + self.dy_prism / 2, self.dy_prism
            )
            self.yprism_topo = np.copy(self.y_prism)
            self.mod_yshape = len(self.y_prism) - 1
        else:
            self.y_prism = np.array([self.yprism_min, self.yprism_max])
            self.mod_yshape = 1
        # Define prisms of initial model
        if self.topo_flag:
            self.mPrism = PP(
                self.earth,
                self.min_size_x,
                self.min_size_y,
                self.min_size_z,
                topo=self.topo_inter,
                dim=self.dim,
                direction=self.direction,
            )
        else:
            self.mPrism = PP(
                self.earth,
                self.min_size_x,
                self.min_size_y,
                self.min_size_z,
                direction=self.direction,
            )
        self.nx_prism = len(self.x_prism) - 1
        self.ny_prism = len(self.y_prism) - 1
        self.nz_prism = len(self.z_prism) - 1
        if self.topo_flag:
            self.nz_prism += 1
        self.n_prisms = self.nx_prism * self.ny_prism * self.nz_prism
        self.prism_nr = []
        self.prism_type = []
        sus = 0.0
        rem = 0.0
        rho = 0.0
        if self.sus_inv:
            sus = 0.001
        if self.rem_inv:
            rem = 0.1
        if self.rho_inv:
            rho = 10.0
        self.zprism_top = 1.0e6
        for i in range(self.mod_xshape):
            xpr = np.array([self.x_prism[i], self.x_prism[i + 1]])
            xtopo = np.array(
                [
                    self.x_prism[i],
                    self.x_prism[i + 1],
                    self.x_prism[i + 1],
                    self.x_prism[i],
                ]
            )
            for j in range(self.mod_yshape):
                ypr = np.array([self.y_prism[j], self.y_prism[j + 1]])
                ytopo = np.array(
                    [
                        self.y_prism[j],
                        self.y_prism[j],
                        self.y_prism[j + 1],
                        self.y_prism[j + 1],
                    ]
                )
                for k in range(self.mod_zshape):
                    if self.topo_flag and k == 0:
                        if self.dim == 3:
                            ztopo = self.topo_inter(xtopo, ytopo)
                        else:
                            zt = self.topo_inter(xtopo)
                            ztopo = np.array([zt[0], zt[0], zt[1], zt[1]])
                        for iz, z in enumerate(ztopo):
                            ztopo[iz] = np.round(z, 1)
                        zt = list(ztopo)
                        zt += 4 * [self.z_prism[0]]
                        zpr = np.array(zt)
                        s = sus / 10.0
                        self.prism_type.append("O")
                        self.prism_layer.append(-1)
                    elif self.topo_flag:
                        zpr = np.array([self.z_prism[k - 1], self.z_prism[k]])
                        self.prism_type.append("P")
                        self.prism_layer.append(k)
                        s = sus
                    else:
                        zpr = np.array([self.z_prism[k], self.z_prism[k + 1]])
                        self.prism_type.append("P")
                        self.prism_layer.append(k)
                        s = sus
                    self.mPrism.add_prism(
                        xpr,
                        ypr,
                        zpr,
                        s,
                        rem,
                        self.earth.inc,
                        self.earth.dec,
                        rho,
                        self.prism_layer[-1],
                        typ=self.data_class.data_type,
                    )
                    self.zprism_top = min(self.zprism_top, zpr.min())
        print(
            "Model prism dictionary defined with " + f"{len(self.mPrism.prisms)} prisms"
        )
        # Prepare book-keeping arrays
        if self.sus_inv and self.rem_inv:
            self.par_hist.append((np.zeros(self.mPrism.n_prisms), 2))
        else:
            self.par_hist.append((np.zeros(self.mPrism.n_prisms), 1))
        self.param_prism += list(range(self.mPrism.n_prisms))
        self.i0 = self.mPrism.n_prisms
        self.n_param = self.mPrism.n_prisms
        self.params = np.zeros(self.n_param + 1)
        i = -1
        if self.sus_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.getsus()
        if self.rem_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.rem
        if self.rho_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.rho
        self.prism_layers = np.unique(np.array(self.prism_layer))
        return True

    def set_prisms_test(self):
        xav = (self.xprism_min + self.xprism_max) / 2.0
        self.x_prism = np.array([xav - self.dx_prism / 2, xav + self.dx_prism / 2])
        self.z_prism = np.array([0.0, self.zprism_max])
        self.mod_xshape = len(self.x_prism) - 1
        self.mod_zshape = len(self.z_prism) - 1
        self.y_prism = np.array([self.yprism_min, self.yprism_max])
        self.mod_yshape = 1
        # Define prisms of initial model
        self.topo_flag = True
        self.mPrism = PP(
            self.earth,
            self.min_size_x,
            self.min_size_y,
            self.min_size_z,
            topo=self.topo_inter,
            dim=self.dim,
        )
        self.n_prisms = 2
        self.prism_nr = []
        self.prism_type = []
        sus = 0.001
        rem = 0.0
        rho = 0.0
        xpr = np.array([self.x_prism[0], self.x_prism[1]])
        ypr = np.array([self.y_prism[0], self.y_prism[1]])
        zt = 4 * [self.z_prism[0]]
        zt += 4 * [self.z_prism[1]]
        zpr = np.array(zt)
        self.prism_type.append("O")
        self.prism_layer.append(-1)
        self.mPrism.add_prism(
            xpr,
            ypr,
            zpr,
            sus,
            rem,
            self.earth.inc,
            self.earth.dec,
            rho,
            self.prism_layer[-1],
            typ=self.data_class.data_type,
        )
        zpr = np.array([self.z_prism[0], self.z_prism[1]])
        self.prism_type.append("P")
        self.prism_layer.append(0)
        self.mPrism.add_prism(
            xpr,
            ypr,
            zpr,
            sus,
            rem,
            self.earth.inc,
            self.earth.dec,
            rho,
            self.prism_layer[-1],
            typ=self.data_class.data_type,
        )
        self.par_hist.append((np.zeros(self.mPrism.n_prisms), 1))
        self.param_prism += list(range(self.mPrism.n_prisms))
        self.i0 = self.mPrism.n_prisms
        self.n_param = self.mPrism.n_prisms
        self.params = np.zeros(self.n_param + 1)
        i = -1
        for key, val in self.mPrism.prisms.items():
            i += 1
            self.params[i] = val.getsus()
        return True

    def get_variances(self):
        """
        Get variances for data and parameters for normalization of inversion
        matrices

        Defines the following variables
        -------------------------------
        sigma_mag, sigma_grav : floats
            Uncertainty of magnetic data [nT] and gravity data [mGal])
        sigma_sus, sigma_rem, sigma_rho : floats
            Variability of susceptibilities [SI], remanence [A/m],
            densities [kg/m3]
        depth_ref : float
            Factor by which to reduce sensitivity at bottom of model with
            respect to top.
            For sus & rem, this regularization factor is cubed.
            For density, it is squared.
        width_max : int
            Number of samples around every data point (to each side) defining
            the area over which the sample value must be larger than the
            neighbours to be considered as local maximum
        max_amp : float
            maximum relative amplitude [%/100] to be considered for a local
            anomaly in order to split prims (if max_val[i]/(max(max_val
            min(max_val)) > max_amp, prisms are split into up to 8 smaller
            prisms (division by 2 in all directions) for the next iteration)

        """
        if self.sus_inv:
            if self.rem_inv:
                results, okButton = dialog(
                    [
                        "sigma_mag",
                        "\n",
                        "sigma_sus",
                        "sigma_rem",
                        "\n",
                        "sigma factor at bottom\n   will be cubed",
                        "width_max [samples]",
                        "max_amp [%]",
                    ],
                    ["e", "l", "e", "e", "l", "e", "e", "e"],
                    [
                        self.sigma_mag,
                        None,
                        self.sigma_sus,
                        self.sigma_rem,
                        None,
                        0.5,
                        self.width_max,
                        self.max_amp * 100.0,
                    ],
                    "Inversion parameters",
                )
                if not okButton:
                    print("\nNo inversion done: no inversion parameters given")
                    return False
                self.sigma_mag = float(results[0])
                self.sigma_sus = float(results[2])
                self.sigma_rem = float(results[3])
                self.depth_ref = float(results[5])
                self.width_max = int(results[6])
                self.max_amp = float(results[7]) / 100.0
            else:
                results, okButton = dialog(
                    [
                        "sigma_mag",
                        "\n",
                        "sigma_sus",
                        "\n",
                        "sigma factor at bottom\n   will be cubed",
                        "width_max [samples]",
                        "max_amp [%]",
                    ],
                    ["e", "l", "e", "l", "e", "e", "e"],
                    [
                        self.sigma_mag,
                        None,
                        self.sigma_sus,
                        None,
                        0.5,
                        self.width_max,
                        self.max_amp * 100.0,
                    ],
                    "Inversion parameters",
                )
                if not okButton:
                    print("\nNo inversion done: no inversion parameters given")
                    return False
                self.sigma_mag = float(results[0])
                self.sigma_sus = float(results[2])
                self.depth_ref = float(results[4])
                self.width_max = int(results[5])
                self.max_amp = float(results[6]) / 100.0
        elif self.rem_inv:
            results, okButton = dialog(
                [
                    "sigma_mag",
                    "\n",
                    "sigma_rem",
                    "\n",
                    "sigma factor at bottom\n   will be squared",
                    "width_max [samples]",
                    "max_amp [%]",
                ],
                ["e", "l", "e", "l", "e", "e", "e"],
                [
                    self.sigma_mag,
                    None,
                    self.sigma_rem,
                    None,
                    0.5,
                    self.width_max,
                    self.max_amp * 100.0,
                ],
                "Inversion parameters",
            )
            if not okButton:
                print("\nNo inversion done: no inversion parameters given")
                return False
            self.sigma_mag = float(results[0])
            self.sigma_rem = float(results[2])
            self.depth_ref = float(results[4])
            self.width_max = int(results[5])
            self.max_amp = float(results[6]) / 100.0
        else:
            results, okButton = dialog(
                [
                    "sigma_grav",
                    "\n",
                    "sigma_rho",
                    "\n",
                    "sigma factor at bottom\n   will be squared",
                    "width_max [samples]",
                    "max_amp [%]",
                ],
                ["e", "l", "e", "l", "e", "e", "e"],
                [
                    self.sigma_grav,
                    None,
                    self.sigma_rho,
                    None,
                    0.5,
                    self.width_max,
                    self.max_amp * 100.0,
                ],
                "Inversion parameters",
            )
            if not okButton:
                print("\nNo inversion done: no inversion parameters given")
                return False
            self.sigma_grav = float(results[0])
            self.sigma_rho = float(results[2])
            self.depth_ref = float(results[4])
            self.width_max = int(results[5])
            self.max_amp = float(results[6]) / 100.0
        return True

    def write_parameters(self, file):
        """
        Save inversion control parameters to file parameters_date&time.dat

        Parameters
        ----------
        file : str
            name of output file

        Returns
        -------
        None.

        """
        with open(file, "w") as fo:
            fo.write(f"{self.data_class.title}\n\n")
            fo.write("File:\n")
            fo.write(f"   {self.data_class.file_name}\n")
            fo.write(
                f"\nEarth's field: strength: {self.earth.f} nT, "
                + f"inclination: {self.earth.inc} deg, "
                + f"declination: {self.earth.dec} deg\n"
            )
            fo.write("\nParameters\n")
            if self.sus_inv:
                fo.write("   Invert for susceptibilities\n")
            if self.rem_inv:
                fo.write("   Invert for remanence\n")
            if self.rho_inv:
                fo.write("   Invert for densities\n")
            fo.write(f"   Maximum number of iterations: {self.max_iter}\n")
            fo.write(
                f"   Stop if relative misfit < {self.max_diff_fac*100}% "
                + "or relative misfit variation < "
                + f"{self.max_rel_diff*100}%\n"
            )
            fo.write(
                f"   Lambda: Initial: {self.lam}, minimum: "
                + f"{self.lam_min}, factor per iteration: {self.lam_fac}\n"
            )
            fo.write(
                f"   Gamma:  Initial: {self.gam}, minimum: {self.gam_min}"
                + f", factor per iteration: {self.gam_fac}\n"
            )
            fo.write("\nInitial prism area\n")
            fo.write(
                f"   West:  {self.xprism_min}, East:  {self.xprism_max}, "
                + f"initial size in X: {self.dx_prism}, "
                + f"min_size_x: {self.min_size_x}\n"
            )
            fo.write(
                f"   South: {self.yprism_min}, North: {self.yprism_max}, "
                + f"initial size in Y: {self.dy_prism}, "
                + f"min_size_y: {self.min_size_y}\n"
            )
            fo.write(
                f"   Top:   {self.zprism_min}, Bottom: {self.zprism_max},"
                + f" initial size in Z: {self.dz_prism}, "
                + f"min_size_z: {self.min_size_z}\n"
            )
            fo.write("\nData uncertainty:\n")
            if "m" in self.data_type:
                fo.write(f"   Magnetics: {self.sigma_mag}\n")
            if self.data_type == "g":
                fo.write(f"   Gravity:   {self.sigma_grav}\n")
            fo.write("\nParameter variability:\n")
            if self.sus_inv:
                fo.write(f"   Susceptibility: {self.sigma_sus}\n")
            if self.rem_inv:
                fo.write(f"   Remancence: {self.sigma_rem}\n")
            if self.rho_inv:
                fo.write(f"   Gravity:   {self.sigma_rho}\n")
            fo.write(
                "   Factor at bottom for weighting of parameter "
                + f"variability: {self.depth_ref}\n"
            )
            fo.write(
                "   Size of area around a point for local maximum "
                + f"determination: {self.width_max} points\n"
            )
            fo.write(
                "   Minimum relative amplitude for prism split: "
                + f"{self.max_amp*100}%\n"
            )
