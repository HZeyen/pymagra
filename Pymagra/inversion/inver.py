# -*- coding: utf-8 -*-
"""
Last modified on Nov 11, 2024

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
from matplotlib.patches import Rectangle
from ..in_out.dialog import dialog
from .potential_prism import Prism_calc as PP
from . import mag_grav_utilities as utils
from ..plotting.new_window import newWindow


class inversion():
    """
    class controlling inversion of data sets
    """

    def __init__(self, data_class, data, x, y, z, topo_int=None, earth=None,
                 data_type="m", line_pos=None, direction=None, dim=3):
        """
        Initialization of inversion.

        Parameters
        ----------
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
        earth: class Earth_mag object, optional. Default=None
            Contains the properties of the Earth's magnetic field
        data_type : str, optional. Default: "m"
            Type of data to be inverted: "m"=magnetic data, "g"=gravity data.
            May be capitals or not.
        line_pos : float, optional. Defuult: None
            For 2.5D inversion, coordinate of the line to be inverted (X or Y).
            Only used for plotting.
        direction : str, optional. Default: None

        """
        self.data_class = data_class
        self.data_shape = data[0].shape
        self.data1_shape = data[0].shape
        self.data = data[0].flatten()
        self.line_pos = line_pos
        self.direction = direction
        self.n_data = len(self.data)
        self.n_data1 = len(self.data)
        self.x = x[0].flatten()
        self.z = z[0].flatten()
        if min(self.x) > 100000.:
            self.dfac = 0.001
        else:
            self.dfac = 1.
        self.topo_int = topo_int
        if topo_int is None:
            self.topo_flag = False
        else:
            self.topo_flag = True
        self.dim = dim
        if y[0] is None:
            self.y = np.zeros_like(self.x)
        else:
            self.y = y[0].flatten()
            # self.dim = 3
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
            self.x2 = x[1].flatten()
            self.z2 = z[1].flatten()
            if self.dim == 3:
                self.y2 = y[1].flatten()
            else:
                self.y2 = np.zeros_like(self.x2)
            self.n_sensor = 2
            # self.sensor2_data = len(self.data2)
            self.n_data2 = len(self.data2)
        self.data_type = data_type.lower()
        self.earth = earth
        self.sus_inv = False
        self.rem_inv = False
        self.rho_inv = False
        self.positive = False
        self.xprism_min = 0.
        self.xprism_max = 0.
        self.dx_prism = 0.
        self.min_size_x = 0.
        self.zprism_min = 0.
        self.zprism_max = 0.
        self.dz_prism = 0.
        self.min_size_z = 0.
        self.yprism_min = 0.
        self.yprism_max = 0.
        self.dy_prism = 0.
        self.min_size_y = 0.
        self.sigma_mag = 1.
        self.sigma_grav = 1.
        self.sigma_sus = 0.001
        self.sigma_rem = 0.04
        self.sigma_rho = 1.
        self.max_iter = 10
        self.max_diff_fac = 0.01
        self.max_rel_diff = 0.001
        if "m" in data_type:
            self.lam = 0.0001
            self.gam = 0.001
        else:
            self.lam = 0.01
            self.gam = 0.01
        self.lam_fac = 0.7
        self.lam_min = 1.E-6
        self.gam_fac = 0.7
        self.gam_min = 1.E-6
        self.sigma_mag = 1.
        self.sigma_grav = 1.
        self.sigma_sus = 1.E-6
        self.sigma_rem = 1.
        self.sigma_rho = 10.
        self.depth_ref = 1.
        self.width_max = 5
        self.max_amp = 0.1
        self.fig_sus = None
        self.fig_rem = None
        self.fig_rho = None
        self.fig_theo = None
        self.fig_theo2 = None
        self.pi4 = np.pi*4.
        self.param_prism = []
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
        self.folder = f"inversion_{self.d1}_{self.c_time}"
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
                    self.sus_inv, self.rem_inv, self.rho_inv, self.x,
                    self.y, self.z)
                self.data_mod = np.matmul(self.G, self.params)
                return
# Calculate effect of actual model
            print(f"\nStart iteration {self.iteration}")

# Create Frechet matrix and covariance matrices
            self.n_param = self.mPrism.get_n_param(
                self.sus_inv, self.rem_inv, self.rho_inv)
            self.S = self.mPrism.create_smooth(
                self.sus_inv, self.rem_inv, self.rho_inv, self.sigma_sus,
                self.sigma_rem, self.sigma_rho, self.depth_ref)
            self.G = self.mPrism.create_Frechet(
                self.sus_inv, self.rem_inv, self.rho_inv, self.x, self.y,
                self.z)
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

            if self.positive:
                fit = []
                self.dat = self.data_ori
                dd = np.copy(self.dat)
                self.params[:-1] = 0.001
                self.params[-1] = 0.
                for it in range(50):
                    self.yparam = np.copy(self.params)
                    self.yparam[:-1] = np.log(self.params[:-1])
                    fac = np.copy(self.params)
                    fac[-1] = 1.
                    G0 = self.G*fac
                    GCT = G0.T*self.sigma_data
                    G_inv = np.matmul(GCT, G0)
                    Gi_max = abs(G_inv).max()
                    Gp = self.lam*self.sigma_param*0.001
                    Gp_max = Gp.max()
                    Gp_fac = Gi_max/Gp_max
                    Gs = self.gam*self.S*0.001
                    Gs_max = Gs.max()
                    Gs_fac = Gi_max/Gs_max
                    if self.iteration == 0 and it == 0:
                        print(f"\nMaximum GT*Cd*G: {Gi_max:0.1f}")
                        print(f"   Maximum regularization: {Gp_max:0.1f} (fac:"
                              + f" {Gp_fac:0.1f})")
                        print(f"   Maximum smoothing     : {Gs_max:0.1f} (fac:"
                              + f" {Gs_fac:0.1f})")
                    G_inv += Gs
                    G_inv[np.diag_indices(G_inv.shape[0])] += Gp
                    d_par = np.matmul(np.matmul(np.linalg.inv(G_inv), GCT), dd)
                    yp = self.yparam+d_par
                    if yp[:-1].max() > 0.:
                        ip = np.argmax(yp[:-1])
                        factor = -self.yparam[ip]/d_par[ip]
                    else:
                        factor = 1.
                    self.yparam[:-1] += d_par[:-1]*factor
                    self.yparam[-1] += d_par[-1]
                    self.params = np.copy(self.yparam)
                    self.params[:-1] = np.exp(self.yparam[:-1])
                    self.params[:-1][self.params[:-1] < 1.E-7] = 1.E-7
                    self.data_mod = np.matmul(self.G, self.params)
                    dd = self.dat - self.data_mod
                    fit.append(np.std(dd))
                    if it < 2:
                        continue
                    if abs((fit[-2] - fit[-1])/fit[-2]) < 1.E-5:
                        break
# Do inversion
            else:
                GCT = self.G.T*self.sigma_data
                G_inv = np.matmul(GCT, self.G)
# For first iteration test whether regularization and smoothing matrices have
#     an appreciable effect. If not, give a warning message
                if self.iteration == 0:
                    mG = abs(G_inv).max()
                    print(f"\nMaximum GT*Cd*G: {mG:0.1f}")
                    mSig = (self.lam*self.sigma_param).max()
                    mSmo = (self.gam*self.S).max()
                    G_Sig = mG/mSig
                    G_Smo = mG/mSmo
                    print(f"   Maximum regularization: {mSig:0.1f} (fac: "
                          + f"{G_Sig:0.1f})")
                    print(f"   Maximum smoothing: {mSmo:0.1f} (fac: "
                          + f"{G_Smo:0.1f})")
                    if G_Sig > 10 or G_Smo > 10:
                        answer = QtWidgets.QMessageBox.warning(
                            None, "Warning",
                            "Regularization or smoothing much smaller than "
                            + "Frechet\n"
                            + f"     Frechet/Regularization: {G_Sig:0.0f}\n"
                            + f"     Frechet/Smoothing: {G_Smo:0.0f}\n"
                            + "You may increase lambda/gamma or restart and "
                            + "decrease parameter variances\n\n"
                            + "Ignore to continue nevertheless\nRetry to give "
                            + "new initial lambda/gamma\nAbort "
                            + "to stop inversion\n",
                            QtWidgets.QMessageBox.Ignore |
                            QtWidgets.QMessageBox.Abort |
                            QtWidgets.QMessageBox.Retry,
                            QtWidgets.QMessageBox.Ignore)
                        if answer == QtWidgets.QMessageBox.Abort:
                            sys.exit()
                        elif answer == QtWidgets.QMessageBox.Retry:
                            results, okButton = dialog(
                                    ["New Initial lambda (regularization)",
                                     "New initial gamma (smoothing)"],
                                    ["e", "e"], [self.lam, self.gam],
                                    "New inversion parameters")
                            if okButton:
                                self.lam = float(results[0])
                                self.gam = float(results[1])

                G_inv[np.diag_indices(G_inv.shape[0])] +=\
                    self.lam*self.sigma_param
                G_inv += self.gam*self.S
                d_par = np.matmul(np.matmul(np.linalg.inv(G_inv), GCT),
                                  self.dat)
                self.params += d_par
            self.data_mod = np.matmul(self.G, self.params)
# Extract new prism properties from parameter vector, set the correspondig
#   values in the prism parameters and copy them into vector par_hist
            i0 = 0
            ss = []
            if self.sus_inv:
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].sus = self.params[i]
                    ss.append(self.mPrism.prisms[key].sus)
                i0 += self.mPrism.n_prisms
            if self.rem_inv:
                sr = []
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].rem = self.params[i+i0]
                    sr.append(self.mPrism.prisms[key].rem)
            if self.sus_inv:
                if self.rem_inv:
                    for i, s in enumerate(ss):
                        self.par_hist.append(np.array([s*self.pi4, sr[i]]))
                else:
                    self.par_hist.append(np.array(ss)*self.pi4)
            elif self.rem_inv:
                self.par_hist.append(sr)
            if self.rho_inv:
                s = []
                for i, key in enumerate(self.mPrism.prisms.keys()):
                    self.mPrism.prisms[key].rho = self.params[i]
                    s.append(self.mPrism.prisms[key].rho)
                self.par_hist.append(np.array(s))
# Calculate magnetic misfits and some statistics
            data = self.data_ori-self.data_mod
            data -= np.nanmedian(data)
            self.data = np.copy(data)
            std_data0 = std_data
            std_data = np.std(data)
            self.std_data_rel = std_data/self.std_data_ori
            std_data_diff = (std_data0-std_data)/self.std_data_ori
            self.RMS_misfit.append(std_data)
            self.rel_RMS_misfit.append(self.std_data_rel*100.)
            if self.std_data_rel < self.max_diff_fac:
                self.stop_RMS_flag = True
            if abs(std_data_diff) < self.max_rel_diff:
                self.stop_Diff_flag = True
            print(f"Maximum data difference data: {abs(data).max()}")
            print(f"Relative Std misfit [%]: {self.std_data_rel*100}")
            print(f"Relative variation of misfit [%]: {std_data_diff*100}")
            print(f"Average: {self.params[-1]}, "
                  + f"sus:({self.params[:-1].min()*1000.}, "
                  + f"{self.params[:-1].max()*1000.})\n")
# If Maximum iteration number is reached, misfit does not become smaller or
# relative misfits are smaller than predefined minimum value, stop iterations
            if self.iteration == self.max_iter:
                print("\nMaximum number of iterations reached\n"
                      + "Iterations stop")
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
            if self.n_sensor == 2:
                _, maxima1 = utils.get_extremes(
                    abs(data[:self.n_data1]).
                    reshape(self.data1_shape), self.width_max)
                _, maxima2 = utils.get_extremes(
                    abs(data[self.n_data1:]).
                    reshape(self.data2_shape), self.width_max)
                maxima = maxima1+maxima2
            else:
                _, maxima = utils.get_extremes(
                    abs(data).reshape(self.data_shape), self.width_max)
            for pos in maxima:
                if len(pos) == 1:
                    i = pos[0]
                else:
                    i = pos[0]*self.data_shape[1] + pos[1]
            key_r = []
            if self.n_sensor == 2:
                key_m1 = self.mPrism.get_max_prisms(
                    abs(data[:self.n_data1]).
                    reshape(self.data1_shape),
                    self.G[:self.n_data1, :self.n_prisms],
                    max_lim=self.max_amp, width=self.width_max)
                key_m2 = self.mPrism.get_max_prisms(
                    abs(data[self.n_data1:]).
                    reshape(self.data2_shape),
                    self.G[self.n_data1:, :self.n_prisms],
                    max_lim=self.max_amp, width=self.width_max)
                key_m = list(np.unique(np.array(key_m1+key_m2)))
            else:
                key_m = self.mPrism.get_max_prisms(
                    abs(data.reshape(self.data_shape)),
                    self.G[:self.n_data1, :self.n_prisms],
                    width=self.width_max)
            if self.sus_inv and self.rem_inv:
                if self.n_sensor == 2:
                    key_r1 = self.mPrism.get_max_prisms(
                        abs(data[:self.n_data1]).
                        reshape(self.data1_shape),
                        self.G[:self.n_data1, self.n_prisms:],
                        max_lim=self.max_amp, width=self.width_max)
                    key_r2 = self.mPrism.get_max_prisms(
                        abs(data[self.n_data1:]).
                        reshape(self.data2_shape),
                        self.G[self.n_data1:, self.n_prisms:],
                        max_lim=self.max_amp, width=self.width_max)
                    key_r = list(np.unique(np.array(key_r1+key_r2)))
                else:
                    key_r = self.mPrism.get_max_prisms(
                        abs(data.reshape(self.data1_shape)),
                        self.G[:self.n_data1, self.n_prisms:],
                        max_lim=self.max_amp, width=self.width_max)
            key_split = list(np.unique(np.array(key_m+key_r)))
# Test whether prisms are marked for splitting
            if len(key_split) == 0:
                print("  Prisms reached size limit\nIteration stops")
                return
# Test whether number of prisms becomes larger than number of data points after
# splitting Prisms will be split into up to 8 pieces, the prism itself is
# eliminated. Therefore the factor 7 in the next line.
            if self.n_prisms+prism_add*len(key_split) > self.n_data:
                print("Inversion stopped: More prisms than data points")
                return
            self.prism_del[self.iteration] = {}
            self.prism_new[self.iteration] = {}
            print(f"  Split prisms {key_split}")
            for _, key in enumerate(key_split):
                self.prism_del[self.iteration][key] =\
                    deepcopy(self.mPrism.prisms[key])
                key_add = self.mPrism.split(key)
                self.prism_new[self.iteration][key] = {}
                for k in key_add:
                    self.prism_new[self.iteration][key][k] =\
                        deepcopy(self.mPrism.prisms[k])
            self.n_prisms = len(list(self.mPrism.prisms.keys()))
            self.params = []
            param_prism = []
            if self.sus_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.sus)
                    param_prism.append(key)
            if self.rem_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.rem)
                    param_prism.append(key)
            if self.rho_inv:
                for key, val in self.mPrism.prisms.items():
                    self.params.append(val.rho)
                    param_prism.append(key)
            self.params.append(0.)
            self.params = np.array(self.params)
            self.lam = max(self.lam*self.lam_fac, self.lam_min)
            self.gam = max(self.gam*self.gam_fac, self.gam_min)
            print(f"New lambda: {self.lam}; new gamma: {self.gam}")
            if self.max_iter == 0:
                return

    def sigmas(self):
        icol = 0
        if self.depth_ref > 1.:
            fac = (self.depth_ref-1.)/self.zprism_max
        elif self.depth_ref < 1.:
            fac = (1./self.depth_ref-1.)/self.zprism_max
        else:
            fac = 1.
        if "m" in self.data_type:
            sigma_data = np.ones(self.n_data)/self.sigma_mag**2
            i = -1
            if self.sus_inv:
                sigma_param = np.ones(self.mPrism.n_prisms)\
                    / self.sigma_sus**2
                if not np.isclose(self.depth_ref, 1.):
                    for _, val in self.mPrism.prisms.items():
                        i += 1
                        f = 1.+(val.z[0]+val.z[1])/2.*fac
                        if self.depth_ref < 1.:
                            f = 1./f
                        sigma_param[i] *= f**3
                icol += self.n_prisms
            elif self.rem_inv:
                sigma_param = np.ones(self.mPrism.n_prisms)\
                    / self.sigma_rem**2
                if not np.isclose(self.depth_ref, 1.):
                    for _, val in self.mPrism.prisms.items():
                        i += 1
                        f = 1.+(val.z[0]+val.z[1])/2.*fac
                        if self.depth_ref < 1.:
                            f = 1./f
                        sigma_param[i] *= f**3
                icol += self.n_prisms
        else:
            sigma_data = np.ones(self.n_data)/self.sigma_grav**2
            sigma_param =\
                np.ones(self.mPrism.n_prisms)/self.sigma_rho**2
            i = -1
            if not np.isclose(self.depth_ref, 1.):
                for _, val in self.mPrism.prisms.items():
                    i += 1
                    f = 1.+(val.z[0]+val.z[1])/2.*fac
                    if self.depth_ref < 1.:
                        f = 1./f
                    sigma_param[i] *= f**2
        sigma_param = np.concatenate((sigma_param, np.array([0.])))
        return sigma_data, sigma_param

    def show_results2D(self, file):
        """
        Plot results of 2D inversion.

        Parameters
        ----------
        file : str
            Base name of file where to store the image

        """
        equal_flag = False
        cmap = plt.get_cmap("rainbow")
        if self.sus_inv:
            txt = "Susceptibility"
            title = "Magnetic data"
            ylabel = "Anomaly [nT]"
            c_txt = "Sus [SI*10**-3]"
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
        if self.direction in ("N", "S", "Y", 0., 180.):
            title += f" at X={self.line_pos*self.dfac}"
        else:
            title += f" at Y={self.line_pos*self.dfac}"
        title += f"; RMS misfit: {self.std_data_rel*100:0.2f}%"
        if file[-1] == "N":
            sens = "Easting"
        else:
            sens = "Northing"
# Plot misfit evolution
        self.fig_mis2 = newWindow(f"{txt} model", 1000, 750, 15, 15)
        self.fig_mis2.fig.tight_layout(w_pad=15, h_pad=2)
        ax_mis = self.fig_mis2.fig.add_subplot()
        ax_mis.plot(np.arange(len(self.std_data_rel)), self.std_data_rel*100.)
        ax_mis.set_title(title, fontsize=14)
        ax_mis.set_xlabel("Iteration #", fontsize=12)
        ax_mis.set_ylabel("Relative RMS misfit [%]", fontsize=12)
        ax_mis.set_xlim([0, len(self.RMS_misfit)-1])
        self.fig_mis2.show()
        fil = os.path.join(self.folder, file+"misfit.png")
        self.fig_mis2.fig.savefig(fil)

        while True:
            self.fig_inv2 = newWindow(f"{txt} model", 2000, 1500, 15, 15)
            self.fig_inv2.fig.tight_layout(w_pad=15, h_pad=2)
            self.gs = GridSpec(16, 10, self.fig_inv2.fig)
            ax_dat = self.fig_inv2.fig.add_subplot(self.gs[1:7, 1:9])
            ax_mod = self.fig_inv2.fig.add_subplot(self.gs[10:16, 1:9])
            xmin = self.x.min()
            xmax = self.x.max()
            if xmax-xmin > 10000:
                xx = self.x/1000.
                xmin /= 1000.
                xmax /= 1000.
                unit = "km"
                fac = 0.001
            else:
                xx = self.x
                unit = "m"
                fac = 1.
            if self.n_sensor == 1:
                dat = self.data_ori+np.nanmedian(self.data_mod)
                ax_dat.plot(xx, dat, "b*", label="meas. data")
                ax_dat.plot(xx, self.data_mod, "cyan", label="calc. data")
            else:
                dat = self.data_ori[:self.n_data1]\
                    + np.nanmedian(self.data_mod[:self.n_data1])\
                    - np.nanmedian(self.data_ori[:self.n_data1])
                ax_dat.plot(xx[:self.n_data1], dat, "b*",
                            label="meas. data sensor 1")
                ax_dat.plot(xx[:self.n_data1], self.data_mod[:self.n_data1],
                            "cyan", label="calc. data sensor 1")
                dat = self.data_ori[self.n_data1:]\
                    + np.nanmedian(self.data_mod[self.n_data1:])\
                    - np.nanmedian(self.data_ori[self.n_data1:])
                ax_dat.plot(xx[self.n_data1:], dat, "r+",
                            label="meas. data sensor 2")
                ax_dat.plot(xx[self.n_data1:], self.data_mod[self.n_data1:],
                            "orange", label="calc. data sensor 2")
                ax_dat.legend(bbox_to_anchor=(1, 1), loc="upper right",
                              fontsize=10)
            ax_dat.tick_params(axis='both', labelsize=12)
            ax_dat.set_title(title, fontsize=14)
            ax_dat.set_xlabel(f"{sens} [{unit}]", fontsize=12)
            ax_dat.set_ylabel(ylabel, fontsize=12)
            ax_dat.set_xlim([xmin, xmax])
            cax = ax_dat.inset_axes([1.05, 0.05, 0.02, 0.9],
                                    transform=ax_dat.transAxes)
            cax.axis("off")
            patches = []
            col = []
            ymin = 1e10
            ymax = -1E10
            for key, val in self.mPrism.prisms.items():
                x1 = val.x[0]*fac
                x2 = val.x[1]*fac
                y1 = val.z[0]*fac
                y2 = val.z[1]*fac
                ymin = min(ymin, y1)
                ymax = max(ymax, y2)
                if self.sus_inv:
                    col.append(val.sus*self.pi4*1000.)
                elif self.rem_inv:
                    col.append(val.rem)
                else:
                    col.append(val.rho)
                patches.append(Rectangle((x1, y1), x2-x1, y2-y1))
            col = np.array(col)
            p = PatchCollection(patches, cmap=cmap)
            p.set_array(col)
            ax_mod.add_collection(p)
            ax_mod.set_xlim([xmin, xmax])
            ax_mod.set_ylim([ymin, ymax])
            ax_mod.invert_yaxis()
            ax_mod.tick_params(axis='both', labelsize=12)
            ax_mod.set_title(txt, fontsize=14)
            ax_mod.set_xlabel(f"{sens} [{unit}]", fontsize=12)
            ax_mod.set_ylabel(f"Depth [{unit}]", fontsize=12)
            if equal_flag:
                ax_mod.set_aspect("equal")
            else:
                ax_mod.set_aspect("auto")
            cax = ax_mod.inset_axes([1.05, 0.05, 0.02, 0.9],
                                    transform=ax_mod.transAxes)
# Plot color bar
            cbar = plt.colorbar(p, orientation="vertical", cax=cax,
                                fraction=0.1)
            cbar.set_label(label=c_txt, size=12)
            for lab in cbar.ax.yaxis.get_ticklabels():
                lab.set_fontsize(12)
            self.fig_inv2.setHelp(
                "Press ENTER to finish; press r to continue iterations; "
                + "press c to toggle between equal scale and filling window "
                + "for model axis. If no reaction, move mouse a little bit")
            self.fig_inv2.show()
# Save inversion control parameters to file parameters.dat
            file_name = os.path.join(self.folder, "parameters.dat")
            self.write_parameters(file_name)
            if equal_flag:
                fil = os.path.join(self.folder, file+"_scaled.png")
            else:
                fil = os.path.join(self.folder, file+"_not-scaled.png")
            self.fig_inv2.fig.savefig(fil)
            while True:
                event = self.fig_inv2.get_event()
                self.fig_inv2.close()
                self.fig_mis2.close()
                if event.name == "key_press_event":
                    if event.key == "enter":
                        return False
                    if event.key in ("c", "C"):
                        equal_flag = not equal_flag
                        break
                    elif event.key in ("r", "R"):
                        return True

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
            txt = "Susceptibility"
            c_txt = "Sus [SI*10**-3]"
        elif self.rem_inv:
            txt = "Remanence"
            c_txt = "Remanence [A/m]"
        else:
            txt = "Density"
            c_txt = "Density [kg/m3]"
# Calculate number of subplots to be done, depending on number of layers
#    calculated
        i0 = 0
        n_prop_plots = len(self.nz_plot)
        max_ax_col = int(np.ceil(np.sqrt(n_prop_plots)))
        max_ax_row = int(np.ceil(n_prop_plots/max_ax_col))
        nax_plot = n_prop_plots-max_ax_col
        npltx = min(max_ax_col, n_prop_plots)
        nplty = max_ax_row
        width = npltx*5
        height = nplty*5
        par = []
# Plot model parameter distribution
        for key, val in self.mPrism.prisms.items():
            if self.sus_inv:
                par.append(val.sus*self.pi4*1000.)
            elif self.rem_inv:
                par.append(val.rem)
            else:
                par.append(val.rho)
        par = np.array(par)
        i0 += self.mPrism.n_prisms
        k = 0
        vmin = np.quantile(par, 0.01)
        vmax = np.quantile(par, 0.99)
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.fig_par = newWindow(f"{txt} model", 2400, 1800, width, height)
        self.fig_par.fig.tight_layout(w_pad=15, h_pad=2)
        self.gs = GridSpec(nplty*10, npltx*10+2, self.fig_par.fig)
        figy0 = -10
        for i in range(nplty):
            figy0 += 10
            figx0 = -10
            for j in range(npltx):
                figx0 += 10
                patches = []
                col = []
                ax = self.fig_par.fig.add_subplot(self.gs[figy0:figy0+8,
                                                          figx0:figx0+8])
                if k >= n_prop_plots:
                    ax.axis("off")
                    k += 1
                    continue
                for key, val in self.mPrism.prisms.items():
                    if val.z[0] <= self.z_plot[k] <= val.z[1]:
                        if self.xprism_min > 100000.:
                            x1 = val.x[0]/1000.
                            x2 = val.x[1]/1000.
                            y1 = val.y[0]/1000.
                            y2 = val.y[1]/1000.
                        else:
                            x1 = val.x[0]
                            x2 = val.x[1]
                            y1 = val.y[0]
                            y2 = val.y[1]
                        if self.sus_inv:
                            col.append(val.sus*self.pi4*1000.)
                        elif self.rem_inv:
                            col.append(val.rem)
                        else:
                            col.append(val.rho)
                        patches.append(Rectangle((x1, y1), x2-x1, y2-y1))
                col = np.array(col)
                p = PatchCollection(patches, cmap="rainbow", norm=norm,
                                    edgecolors=("black",))
                p.set_array(col)
                ax.add_collection(p)
                ax.set_title(f"{txt} at {self.z_plot[k]:0.1f} m", fontsize=10)
                ax.tick_params(top=True, right=True, labelsize=10)
                if k+npltx+1 > n_prop_plots:
                    ax.set_xlabel(f"Easting [{self.ax_unit}]", fontsize=10)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(f"Northing [{self.ax_unit}]", fontsize=10)
                elif j == npltx-1 or k == n_prop_plots-1:
                    ax.yaxis.tick_right()
                else:
                    ax.set_yticklabels([])
                ax.set_xlim([self.xprism_min, self.xprism_max])
                ax.set_ylim([self.yprism_min, self.yprism_max])
                ax.set_aspect("equal", adjustable="box")
                if k < nax_plot:
                    ax.set_xlabel("")
                if j > 0:
                    ax.set_ylabel("")
                k += 1
        ax = self.fig_par.fig.add_subplot(self.gs[1:-1, -1:])
        ax.axis("off")
        cax = ax.inset_axes([0., 0., 0.9, 0.9], transform=ax.transAxes)
        cbar = plt.colorbar(p, orientation="vertical", cax=cax, fraction=0.9)
        cbar.set_label(c_txt, fontsize=10)
        cbar.ax.tick_params(labelsize=10)

        self.fig_par.setHelp("Press ENTER to finish, R to continue iterations")
        self.fig_par.show()
        self.write_parameters(os.path.join(self.folder, "parameters.dat"))
        self.fig_par.fig.savefig(
            os.path.join(self.folder, f"{txt}_distribution.png"))

        if "m" in self.data_type:
            self.fig_theo = newWindow("Magnetic data", 1500, 1000, 20, 15)
            self.fig_theo.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Mag_sensor1_calc&diff.png")
            title = "Modelled magnetic sensor1"
            unit = "nT"
            title2 = "Mag difference sensor1\nrel. RMS misfit: "\
                + f"{self.std_data_rel*100:0.2f}%"
        else:
            self.fig_theo = newWindow("Gravity data", 1500, 1000, 20, 15)
            self.fig_theo.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = os.path.join(self.folder, "Gravi_calc&diff.png")
            title = "Modelled gravity data"
            unit = "mGal"
            title2 = "Gravity difference\nrel. RMS misfit: "\
                + f"{self.std_data_rel*100:0.2f}%"
        self.ax_theo = []
        ddx = self.xplt_max - self.xplt_min
        ddy = self.yplt_max - self.yplt_min
        if ddx > 1.5*ddy:
            self.gs = GridSpec(18, 10, self.fig_theo)
            self.ax_theo.append(
                self.fig_theo.fig.add_subplot(self.gs[1:8, 1:]))
            self.ax_theo.append(
                self.fig_theo.fig.add_subplot(self.gs[10:17, 1:]))
            self.bar_or = "vertical"
#                    anchor = 'E'
            self.nticks = 10
# Horizontal layout
        else:
            self.gs = GridSpec(10, 18, self.fig_theo)
            self.ax_theo.append(
                self.fig_theo.fig.add_subplot(self.gs[1:, 1:8]))
            self.ax_theo.append(
                self.fig_theo.fig.add_subplot(self.gs[1:, 10:17]))
            self.bar_or = "horizontal"
#                    anchor = 'S'
            self.nticks = 5

# Plot magnetic anomalies produced by inverted model
        data = self.data_mod[:self.n_data1].reshape(self.data1_shape)
        med = np.nanmedian(data)
        data -= med
        vmin = np.ceil(np.quantile(data, 0.005)*1000)/1000
        vmax = np.ceil(np.quantile(data, 0.995)*1000)/1000
        br_map, norm = utils.mag_color_map(vmin, vmax)
        im, cbar = utils.data_plot(
            data, self.fig_theo.fig, self.ax_theo[0],
            title=f"{title}\nMedian: {med:0.1f}",
            xtitle=f"Easting [{self.ax_unit}]",
            ytitle=f"Northing [{self.ax_unit}]",
            cmap=br_map, norm=norm, cbar_title=unit,
            extent=[self.xplt_min, self.xplt_max,
                    self.yplt_min, self.yplt_max])
# plot prism contours
        for key, val in self.mPrism.prisms.items():
            x1 = val.x[0]
            x2 = val.x[1]
            y1 = val.y[0]
            y2 = val.y[1]
            self.ax_theo[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                                 "k", linewidth=1)
        self.ax_theo[0].set_xlim([self.xplt_min, self.xplt_max])
        self.ax_theo[0].set_ylim([self.yplt_min, self.yplt_max])
        self.ax_theo[0].grid(visible=True, which="both")
        self.ax_theo[0].set_xlabel("")
# Plot misfit
        data = (self.data_mod[:self.n_data1] -
                self.data_ori[:self.n_data1]).reshape(self.data1_shape)
        data -= np.nanmedian(data)
        vmin = np.ceil(np.quantile(data, 0.005)*1000)/1000
        vmax = np.floor(np.quantile(data, 0.995)*1000)/1000
        br_map, norm = utils.mag_color_map(vmin, vmax)
        im, cbar = utils.data_plot(
            data, self.fig_theo.fig, self.ax_theo[1], title=f"{title2}",
            xtitle=f"Easting [{self.ax_unit}]",
            ytitle=f"Northing [{self.ax_unit}]",
            extent=[self.xplt_min, self.xplt_max,
                    self.yplt_min, self.yplt_max], cmap=br_map,
            cbar_title=f"Difference calc-meas [{unit}]", norm=norm)
        self.ax_theo[1].grid(visible=True, which="both")
        self.fig_theo.show()
        self.fig_theo.fig.savefig(plt_file)

# Plot results for sensor 2
        if self.n_sensor == 2:
            self.fig_theo2 = newWindow("Magnetic data", 1500, 1000, 24, 18)
            self.fig_theo2.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file2 = os.path.join(self.folder, "Mag_sensor2_calc&diff.png")
            title2 = "Mag difference sensor2, rel. RMS misfit: "\
                + f"{self.std_data_rel*100:0.2f}%"
            self.ax_theo2 = []
            if ddx > 1.5*ddy:
                self.gs = GridSpec(18, 10, self.fig_theo2)
                self.ax_theo2.append(
                    self.fig_theo2.fig.add_subplot(self.gs[1:8, 1:]))
                self.ax_theo2.append(
                    self.fig_theo2.fig.add_subplot(self.gs[10:17, 1:]))
                self.bar_or = "vertical"
#                    anchor = 'E'
                self.nticks = 10
# Horizontal layout
            else:
                self.gs = GridSpec(10, 18, self.fig_theo2)
                self.ax_theo2.append(
                    self.fig_theo2.fig.add_subplot(self.gs[1:, 1:8]))
                self.ax_theo2.append(
                    self.fig_theo2.fig.add_subplot(self.gs[1:, 10:17]))
                self.bar_or = "horizontal"
#                    anchor = 'S'
                self.nticks = 5
# Plot magnetic anomalies produced by inverted model
            data = self.data_mod[self.n_data1:].reshape(self.data2_shape)
            med = np.median(data)
            data -= med
            vmin = np.ceil(np.quantile(data, 0.005)*1000)/1000
            vmax = np.ceil(np.quantile(data, 0.995)*1000)/1000
            br_map, norm = utils.mag_color_map(vmin, vmax)
            im, cbar = utils.data_plot(
                data, self.fig_theo2.fig, self.ax_theo2[0],
                title=f"Modelled magnetic sensor2\nMedian: {med:0.1f}",
                xtitle=f"Easting [{self.ax_unit}]",
                ytitle=f"Northing [{self.ax_unit}]",
                cmap=br_map, norm=norm, cbar_title=unit,
                extent=[self.xplt_min, self.xplt_max,
                        self.yplt_min, self.yplt_max])
# plot prism contours
            for key, val in self.mPrism.prisms.items():
                x1 = val.x[0]
                x2 = val.x[1]
                y1 = val.y[0]
                y2 = val.y[1]
                self.ax_theo2[0].plot([x1, x2, x2, x1, x1],
                                      [y1, y1, y2, y2, y1], "k", linewidth=1)
            self.ax_theo2[0].set_xlim([self.xplt_min, self.xplt_max])
            self.ax_theo2[0].set_ylim([self.yplt_min, self.yplt_max])
            self.ax_theo2[0].grid(visible=True, which="both")
            self.ax_theo2[0].set_xlabel("")

# Plot misfit
            data = (self.data_mod[self.n_data1:] -
                    self.data_ori[self.n_data1:]).\
                reshape(self.data2_shape)
            data -= np.nanmedian(data)
            vmin = np.ceil(np.quantile(data, 0.005)*1000)/1000
            vmax = np.floor(np.quantile(data, 0.995)*1000)/1000
            br_map, norm = utils.mag_color_map(vmin, vmax)
            im, cbar = utils.data_plot(
                data, self.fig_theo2.fig, self.ax_theo2[1],
                title="Mag difference sensor2\n"
                + f"rel. RMS misfit: {self.std_data_rel*100:0.2f}%",
                xtitle=f"Easting [{self.ax_unit}]",
                ytitle=f"Northing [{self.ax_unit}]",
                extent=[self.xplt_min, self.xplt_max,
                        self.yplt_min, self.yplt_max],
                cbar_title="Difference calc-meas [nT]", cmap=br_map, norm=norm)
            self.ax_theo2[1].grid(visible=True, which="both")
            self.fig_theo2.show()
            self.fig_theo2.fig.savefig(plt_file2)

# Plot misfit evolution
        self.fig_misfit = newWindow("Magnetic data", 800, 500)
        self.ax_misfit = self.fig_misfit.fig.subplots(1, 1)
        self.ax_misfit.plot(np.arange(len(self.rel_RMS_misfit)),
                            self.rel_RMS_misfit, "k")
        self.ax_misfit.set_title("Misfit evolution")
        self.ax_misfit.set_xlabel("Iteration number")
        self.ax_misfit.set_ylabel("RMS misfit [%]")
        self.fig_misfit.show()
        self.fig_misfit.fig.savefig(os.path.join(self.folder,
                                                 "misfit-evolution.png"))
        print("\nClick into parameter distribution window to finish inversion")
        while True:
            event = self.fig_par.get_event()
            if event.name == "key_press_event":
                if event.key in ("enter", "r", "R"):
                    break
        self.fig_par.close()
        self.fig_theo.close()
        self.fig_misfit.close()
        if self.n_sensor == 2:
            self.fig_theo2.close()
        if event.key == "enter":
            return False
        else:
            return True

    def show_synthetic(self):
        """
        Plots results from synthetic model calculation

        """
        if "m" in self.data_type:
            self.fig_syn = newWindow("Magnetic data", 1500, 1000, 20, 15)
            self.fig_syn.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = "Mag_synthetic.png"
            title = f"Modelled magnetic anomaly {self.z[0]}m"
            if self.n_sensor == 2:
                title2 = f"Modelled magnetic anomaly {self.z2[0]}m"
            unit = "nT"
        else:
            self.fig_syn = newWindow("Gravity data", 1500, 1000, 20, 15)
            self.fig_syn.fig.tight_layout(w_pad=15, h_pad=2)
            plt_file = "Gravi_synthetic.png"
            title = "Modelled gravity anomaly"
            unit = "mGal"
# Plot magnetic anomalies produced by synthetic model
        data = self.data_mod[:self.n_data1].reshape(self.data1_shape)
        if self.n_sensor == 2:
            data2 = self.data_mod[self.n_data1:].reshape(self.data2_shape)
            ddx = self.x.max() - self.x.min()
            ddy = self.y.max() - self.y.min()
            facx = 10 / (2 * ddx)
            facy = 8 / (2 * ddy)
# Vertical layout
            self.ax_syn = []
            if facx < facy:
                self.gs = GridSpec(18, 10, self.fig_syn.fig)
                self.ax_syn.append(self.fig_syn.fig.add_subplot(
                    self.gs[1:8, 1:]))
                self.ax_syn.append(self.fig_syn.fig.add_subplot(
                    self.gs[10:17, 1:]))
                self.bar_or = "vertical"
                self.nticks = 10
# Horizontal layout
            else:
                self.gs = GridSpec(10, 18, self.fig_syn.fig)
                self.ax_syn.append(self.fig_syn.fig.add_subplot(
                    self.gs[1:, 1:8]))
                self.ax_syn.append(self.fig_syn.fig.add_subplot(
                    self.gs[1:, 10:17]))
                self.bar_or = "horizontal"
                self.nticks = 5
        else:
            self.gs = GridSpec(10, 10, self.fig_syn.fig)
            self.ax_syn = [self.fig_syn.fig.add_subplot(self.gs[1:-1, 1:-1])]
        vmin = np.ceil(np.quantile(data, 0.005)*1000)/1000
        vmax = np.ceil(np.quantile(data, 0.995)*1000)/1000
        self.xplt_min = self.x.min()
        self.xplt_max = self.x.max()
        self.yplt_min = self.y.min()
        self.yplt_max = self.y.max()
        if "m" in self.data_type:
            br_map, norm = utils.mag_color_map(vmin, vmax)
            im, cbar = utils.data_plot(
                data, self.fig_syn.fig, self.ax_syn[0], title=f"{title}",
                xtitle="Easting [m]",
                ytitle="Northing [m]",
                cmap=br_map, norm=norm, cbar_title=unit,
                extent=[self.xplt_min, self.xplt_max,
                        self.yplt_min, self.yplt_max])
            if self.n_sensor == 2:
                im, cbar = utils.data_plot(
                    data2, self.fig_syn.fig, self.ax_syn[1], title=f"{title2}",
                    xtitle="Easting [m]",
                    ytitle="Northing [m]",
                    cmap=br_map, norm=norm, cbar_title=unit,
                    extent=[self.xplt_min, self.xplt_max,
                            self.yplt_min, self.yplt_max])
        else:
            im, cbar = utils.data_plot(
                data, self.fig_syn.fig, self.ax_syn[0], title=f"{title}",
                xtitle="Easting [m]",
                ytitle="Northing [m]",
                cmap="rainbow", cbar_title=unit,
                extent=[self.xplt_min, self.xplt_max,
                        self.yplt_min, self.yplt_max])
# plot prism contours
        for key, val in self.mPrism.prisms.items():
            x1 = val.x[0]
            x2 = val.x[1]
            y1 = val.y[0]
            y2 = val.y[1]
            self.ax_syn[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                                "k", linewidth=1)
            if self.n_sensor == 2:
                self.ax_syn[1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1],
                                    "k", linewidth=1)
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
        self.fig_syn.close()

    def save_model(self):
        """
        Write model parameters into file model_date&time.dat

        Parameters
        ----------
        file_name : str
            Name of ASCII file where to store the model parameters
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
            fo.write(f"           Prism_center      {text1}                   "
                     + " Prism coordinates\n")
            fo.write(f"      X          Y       Z   {text2}   #       W       "
                     + "  E          S          N      top   bottom\n")
            for key, val in self.mPrism.prisms.items():
                x1 = val.x[0]
                x2 = val.x[1]
                xc = (x1+x2)*0.5
                y1 = val.y[0]
                y2 = val.y[1]
                yc = (y1+y2)*0.5
                z1 = val.z[0]
                z2 = val.z[1]
                zc = (z1+z2)*0.5
                fo.write(f"{xc:9.2f} {yc:10.2f} {zc:7.2f} ")
                if self.sus_inv:
                    fo.write(f"{val.sus*4*np.pi*1E6:10.0f}")
                if self.rem_inv:
                    fo.write(f"{val.rem:7.3f}")
                if self.rho_inv:
                    fo.write(f"{val.rho:7.0f}")
                fo.write(f"{key:6d} {x1:9.2f} {x2:9.2f} {y1:10.2f} {y2:10.2f} "
                         + f"{z1:7.2f} {z2:7.2f}\n")
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
        if "m" in data_type:
            results, okButton = dialog(
                    ["Invert for:",
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
                     "Use positivity constraint"],
                    ["l", "c", "c", "e", "e", "e", "e", "e", "e", "e", "e",
                     "e", "c"],
                    [None, 1, 0, self.max_iter, self.max_diff_fac,
                     self.max_rel_diff, self.lam, self.lam_fac, self.lam_min,
                     self.gam, self.gam_fac, self.gam_min, None],
                    "Magnetic inversion parameters")
        else:
            results, okButton = dialog(
                    ["Maximum number of iterations",
                     "Maximum relative RMS misfit",
                     "Maximum variation of RMS",
                     "Initial lambda (regularization)",
                     "Lambda factor per iteration",
                     "Minimum lambda",
                     "Initial Gamma (smoothing)",
                     "Gamma factor per iteration",
                     "Minimum Gamma",
                     "Use positivity constraint"],
                    ["e", "e", "e", "e", "e", "e", "e", "e", "e", "c"],
                    [self.max_iter, self.max_diff_fac, self.max_rel_diff,
                     self.lam, self.lam_fac, self.lam_min,
                     self.gam, self.gam_fac, self.gam_min, None],
                    "Gravity inversion parameters")
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
                print("\nInverting for both, susceptibility and remanence, "
                      + "does not make sense.\n\nSusceptibility inversion "
                      + "canceled and susceptibility set to zero.")
                self.sus_inv = False
            if not self.sus_inv and not self.rem_inv:
                print("\nNo parameter type for inversion chosen.\n"
                      + "Susceptibility will be used by default")
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
        dx = line_length/(len(self.x)-1)
        nr = 1-int(np.log10(line_length))
        while True:
            dx_ini = np.round(line_length/5., nr)
            if dx_ini > 0.:
                break
            nr += 1
        dz_ini = dx_ini/2.
        zmin = 0
        zmax = dz_ini
        yw = zmax

        nx = int(np.ceil(line_length/dx_ini))
        if dx_ini*nx-dx <= line_length:
            nx += 1
        xmin_p = xmin - zmax
        xmax_p = xmax + zmax

        results, okButton = dialog(
                ["xmin [m]", "xmax [m]", "dx_ini [m]", "minimum size x [m]",
                 "zmin [m]", "zmax [m]", "dz_ini [m]", "minimum size z [m]",
                 "prism_width [m]",
                 "Data point reduction\n(1 point out of...)"],
                ["e", "e", "e", "e", "e", "e", "e", "e", "e", "e"],
                [xmin_p, xmax_p, dx_ini, dx_ini/4., zmin, zmax, dz_ini,
                 dz_ini/4., yw, 1], "2D prism parameters")
        if not okButton:
            print("\nNo prism data given, inversion aborted")
            return False
        self.xprism_min = float(results[0])
        self.xprism_max = float(results[1])
        self.dx_prism = float(results[2])
        self.min_size_x = float(results[3])
        self.zprism_min = float(results[4])
        self.zprism_max = float(results[5])
        self.dz_prism = float(results[6])
        self.min_size_z = float(results[7])
        self.data_reduction = int(results[9])
        yw = float(results[8])
        self.yprism_min = -yw
        self.yprism_max = yw
        self.dy_prism = 2.*yw
        self.min_size_y = self.dy_prism
        self.prepare_data()
        self.set_prisms()
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
        dx = line_length_x/(len(self.x)-1)
        ymin = self.y.min()
        ymax = self.y.max()
        line_length_y = ymax - ymin
        dy = line_length_y/(len(self.y)-1)
        length_xy = max((line_length_x, line_length_y))
        nr = 1-int(np.log10(length_xy))
        dx_ini = np.round(np.sqrt(line_length_x*line_length_y/25.), nr)
        dy_ini = dx_ini
        dz_ini = length_xy/10.
        zmin = 0.
        zmax = dz_ini
        nx = int(np.ceil(line_length_x/dx_ini))
        if dx_ini*nx-dx <= line_length_x:
            nx += 1
        xmin_p = xmin - dx_ini
        xmax_p = xmax + dx_ini

        ny = int(np.ceil(dy/dy_ini))
        if dy_ini*ny-dy <= line_length_y:
            ny += 1
        ymin_p = ymin - dy_ini
        ymax_p = ymax + dy_ini

# next lines are for testing purpose only
        # xmin_p = 680000.
        # xmax_p = 740000.
        # dx_ini = 15000.
        # ymin_p = 2240000.
        # ymax_p = 2285000.
        # dy_ini = 15000.
        # zmax = 5000.
        # dz_ini = 2500.

# Get initial prism configuration and prism control parameters
        results, okButton = dialog(
                ["xmin [m]", "xmax [m]", "dx_ini [m]", "minimum size x [m]",
                 "ymin [m]", "ymax [m]", "dy_ini [m]", "minimum size y [m]",
                 "zmin [m]", "zmax [m]", "dz_ini [m]", "minimum size z [m]",
                 "Data point reduction\n(1 point out of...)"],
                ["e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e", "e",
                 "e"],
                [xmin_p, xmax_p, dx_ini, dx_ini/4, ymin_p, ymax_p, dy_ini,
                 dy_ini/4, zmin, zmax, dz_ini, dz_ini/4, 1],
                "Prism parameters")
        if not okButton:
            print("\nNo prism data given, inversion aborted")
            return False
        self.xprism_min = float(results[0])
        self.xprism_max = float(results[1])
        self.dx_prism = float(results[2])
        self.min_size_x = float(results[3])
        self.yprism_min = float(results[4])
        self.yprism_max = float(results[5])
        self.dy_prism = float(results[6])
        self.min_size_y = float(results[7])
        self.zprism_min = float(results[8])
        self.zprism_max = float(results[9])
        self.dz_prism = float(results[10])
        self.min_size_z = float(results[11])
        self.data_reduction = int(results[12])
# Extract data within prism area and, if asked for, reduce point density
        self.prepare_data()
# Back up original (reduced) data
        self.data_ori = np.copy(self.data)
        self.std_data_ori = np.std(self.data)
        self.xmin = np.nanmin((self.x))
        self.xmax = np.nanmax((self.x))
        self.ymin = np.nanmin((self.y))
        self.ymax = np.nanmax((self.y))
        self.dx = (xmax-xmin)/(self.data_shape[1]-1)
        self.dy = (ymax-ymin)/(self.data_shape[0]-1)
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
            xdata_min = self.xprism_min + self.zprism_max
            xdata_max = self.xprism_max - self.zprism_max
            n1 = np.where(self.x[:ndata] >= xdata_min)[0][0]
            n2 = np.where(self.x[:ndata] <= xdata_max)[0][-1]+1
            for i in range(n1, n2):
                if np.isfinite(self.data[i]):
                    break
            n1 = i
            for i in range(n2-1, n1-1, -1):
                if np.isfinite(self.data[i]):
                    break
            n2 = i+1
            n1 = max(n1, int(self.data_reduction/2))
            x = self.x[n1:n2:self.data_reduction]
            y = self.y[n1:n2:self.data_reduction]
            z = self.z[n1:n2:self.data_reduction]
            data = self.data[n1:n2:self.data_reduction]
            self.data_shape = data.shape
            self.data1_shape = data.shape
            self.n_data1 = len(data)
            if self.n_sensor == 2:
                n1 = np.where(self.x2[:ndata] >= xdata_min)[0][0]
                n2 = np.where(self.x2[:ndata] <= xdata_max)[0][-1]+1
                for i in range(n1, n2):
                    if np.isfinite(self.data2[i]):
                        break
                n1 = i
                for i in range(n2-1, n1-1, -1):
                    if np.isfinite(self.data2[i]):
                        break
                n2 = i+1
                n1 = max(n1, int(self.data_reduction/2))
                x = np.concatenate((x, self.x2[n1:n2:self.data_reduction]))
                y = np.concatenate((y, self.y2[n1:n2:self.data_reduction]))
                z = np.concatenate((z, self.z2[n1:n2:self.data_reduction]))
                d2 = self.data2[n1:n2:self.data_reduction]
                self.n_data2 = len(d2)
                self.data2_shape = d2.shape
                data = np.concatenate((data, d2))
            else:
                self.n_data2 = 0
        else:
            ndata_y, ndata_x = self.data_shape
            ndata = ndata_x*ndata_y
            xdata_min = self.xprism_min + self.zprism_max
            xdata_max = self.xprism_max - self.zprism_max
            ydata_min = self.yprism_min + self.zprism_max
            ydata_max = self.yprism_max - self.zprism_max
            d = self.data.reshape(self.data_shape)
            x = self.x.reshape(self.data_shape)
            y = self.y.reshape(self.data_shape)
            z = self.z.reshape(self.data_shape)
            xcol = np.unique(self.x)
            yrow = np.unique(self.y)
            nx1 = np.where(xcol >= xdata_min)[0][0]
            nx2 = np.where(xcol <= xdata_max)[0][-1]+1
            nx1 = max(nx1, int(self.data_reduction/2))
            ny1 = np.where(yrow >= ydata_min)[0][0]
            ny2 = np.where(yrow <= ydata_max)[0][-1]+1
            ny1 = max(ny1, int(self.data_reduction/2))
            x = x[ny1:ny2:self.data_reduction, nx1:nx2:self.data_reduction].\
                flatten()
            y = y[ny1:ny2:self.data_reduction, nx1:nx2:self.data_reduction].\
                flatten()

            z = z[ny1:ny2:self.data_reduction, nx1:nx2:self.data_reduction].\
                flatten()

            data = d[ny1:ny2:self.data_reduction, nx1:nx2:self.data_reduction]
            self.data_shape = data.shape
            self.data1_shape = data.shape
            data = data.flatten()
            self.n_data1 = len(data)
            if self.n_sensor == 2:
                d = self.data2.reshape(self.data2_shape)
                x2 = self.x2.reshape(self.data2_shape)
                y2 = self.y2.reshape(self.data2_shape)
                z2 = self.z2.reshape(self.data2_shape)
                xcol = np.unique(self.x2)
                yrow = np.unique(self.y2)
                nx1 = np.where(xcol >= xdata_min)[0][0]
                nx2 = np.where(xcol <= xdata_max)[0][-1]+1
                nx1 = max(nx1, int(self.data_reduction/2))
                ny1 = np.where(yrow >= ydata_min)[0][0]
                ny2 = np.where(yrow <= ydata_max)[0][-1]+1
                ny1 = max(ny1, int(self.data_reduction/2))
                x = np.concatenate(
                    (x, x2[ny1:ny2:self.data_reduction,
                           nx1:nx2:self.data_reduction].flatten()))
                y = np.concatenate(
                    (y, y2[ny1:ny2:self.data_reduction,
                           nx1:nx2:self.data_reduction].flatten()))
                z = np.concatenate(
                    (z, z2[ny1:ny2:self.data_reduction,
                           nx1:nx2:self.data_reduction].flatten()))
                d2 = d[ny1:ny2:self.data_reduction,
                       nx1:nx2:self.data_reduction]
                self.data2_shape = d2.shape
                data = np.concatenate((data, d2.flatten()))
                self.n_data2 = d2.shape[0] * d2.shape[1]
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
        self.xplt_min = self.x.min()-self.dx*0.5
        self.xplt_max = self.x.max()+self.dx*0.5
        self.yplt_min = self.y.min()-self.dy*0.5
        self.yplt_max = self.y.max()+self.dy*0.5
        if max(self.xprism_max, self.yprism_max) > 100000:
            self.xplt_min /= 1000.
            self.xplt_max /= 1000.
            self.yplt_min /= 1000.
            self.yplt_max /= 1000.
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
        self.z_plot = np.arange(self.zprism_min+self.min_size_z/2,
                                self.zprism_max, self.min_size_z)
        self.nz_plot = []
        for i, zz in enumerate(self.z_plot):
            self.nz_plot.append(int((zz-self.zprism_min)/self.min_size_z))

    def set_prisms(self):
        """
        Define prisms for initial model

        Returns
        -------
        None.

        """
# Define initial model
        self.x_prism = np.arange(self.xprism_min,
                                 self.xprism_max+self.dx_prism/2,
                                 self.dx_prism)
        self.z_prism = np.arange(self.zprism_min,
                                 self.zprism_max+self.dz_prism/2,
                                 self.dz_prism)
        self.mod_xshape = len(self.x_prism)-1
        self.mod_zshape = len(self.z_prism)-1
        if self.dim == 3:
            self.y_prism = np.arange(self.yprism_min,
                                     self.yprism_max+self.dy_prism/2,
                                     self.dy_prism)
            self.yprism_topo = np.copy(self.y_prism)
            self.mod_yshape = len(self.y_prism)-1
        else:
            self.y_prism = np.array([self.yprism_min, self.yprism_max])
            self.mod_yshape = 1
            # self.y_prism_topo = self.y_prism+
            # self.mod_yshape = 1dd
# Define prisms of initial model
        self.mPrism = PP(self.earth, self.min_size_x, self.min_size_y,
                         self.min_size_z)
        self.nx_prism = len(self.x_prism)-1
        self.ny_prism = len(self.y_prism)-1
        self.nz_prism = len(self.z_prism)-1
        self.n_prisms = self.nx_prism*self.ny_prism*self.nz_prism
        self.prism_nr = []
        if self.topo_flag:
            for i in range(self.mod_xshape):
                xpr = np.array([self.x_prism[i], self.x_prism[i+1]])
                xtopo = np.array([self.x_prism[i], self.x_prism[i+1],
                                  self.x_prism[i+1], self.x_prism[i]])
                for j in range(self.mod_yshape):
                    ypr = np.array([self.y_prism[j], self.y_prism[j+1]])
                    ytopo = np.array([self.y_prism[j], self.y_prism[j],
                                      self.y_prism[j+1],  self.y_prism[j+1]])
                    ztopo = self.topo_int(xtopo, ytopo)
                    ztmin = ztopo.max()
                    z1 = self.zprism_max
                    while True:
                        z2 = z1-self.dz_prism
                        if z2 < ztmin:
                            zpr = np.array([z2, z1])
                        else:
                            zpr = np.array([list(ztopo) + [z1]])
                        z1 -= self.dz_prism
        else:
            for i in range(self.mod_xshape):
                xpr = np.array([self.x_prism[i], self.x_prism[i+1]])
                for j in range(self.mod_yshape):
                    ypr = np.array([self.y_prism[j], self.y_prism[j+1]])
                    for k in range(self.mod_zshape):
                        zpr = np.array([self.z_prism[k], self.z_prism[k+1]])
                        self.mPrism.add_prism(xpr, ypr, zpr, 0.001, 0.1,
                                              self.earth.inc, self.earth.dec,
                                              10.)
        print("Model prism dictionary defined with "
              + f"{len(self.mPrism.prisms)} prisms")
# Prepare book-keeping arrays
        if self.sus_inv and self.rem_inv:
            self.par_hist.append((np.zeros(self.mPrism.n_prisms), 2))
        else:
            self.par_hist.append((np.zeros(self.mPrism.n_prisms), 1))
        self.param_prism += list(range(self.mPrism.n_prisms))
        self.i0 = self.mPrism.n_prisms
        self.n_param = self.mPrism.n_prisms
        self.params = np.zeros(self.n_param+1)
        i = -1
        if self.sus_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.sus
        if self.rem_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.rem
        if self.rho_inv:
            for key, val in self.mPrism.prisms.items():
                i += 1
                self.params[i] = val.rho

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
            Reference depth for decrease of sensitivity of prism parameters.
            For sus & rem, sensitivity (regularization) decreases with
            (depth_ref/depth)**3.
            For density, sensitivity (regularization) decreases with
            (depth_ref/depth)**2.
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
                        ["sigma_mag",
                         "\n",
                         "sigma_sus",
                         "sigma_rem",
                         "\n",
                         "sigma factor at bottom\n   will be cubed",
                         "width_max [samples]",
                         "max_amp [%]"],
                        ["e", "l", "e", "e", "l", "e", "e", "e"],
                        [self.sigma_mag, None, self.sigma_sus, self.sigma_rem,
                         None, 2., self.width_max,
                         self.max_amp*100.], "Inversion parameters")
                if not okButton:
                    print("\nNo inversion done: no inversion parameters given")
                    return False
                self.sigma_mag = float(results[0])
                self.sigma_sus = float(results[2])
                self.sigma_rem = float(results[3])
                self.depth_ref = float(results[5])
                self.width_max = int(results[6])
                self.max_amp = float(results[7])/100.
            else:
                results, okButton = dialog(
                        ["sigma_mag",
                         "\n",
                         "sigma_sus",
                         "\n",
                         "sigma factor at bottom\n   will be cubed",
                         "width_max [samples]",
                         "max_amp [%]"],
                        ["e", "l", "e", "l", "e", "e", "e"],
                        [self.sigma_mag, None, self.sigma_sus, None,
                         2., self.width_max, self.max_amp*100.],
                        "Inversion parameters")
                if not okButton:
                    print("\nNo inversion done: no inversion parameters given")
                    return False
                self.sigma_mag = float(results[0])
                self.sigma_sus = float(results[2])
                self.depth_ref = float(results[4])
                self.width_max = int(results[5])
                self.max_amp = float(results[6])/100.
        elif self.rem_inv:
            results, okButton = dialog(
                    ["sigma_mag",
                     "\n",
                     "sigma_rem",
                     "\n",
                     "sigma factor at bottom\n   will be squared",
                     "width_max [samples]",
                     "max_amp [%]"],
                    ["e", "l", "e", "l", "e", "e", "e"],
                    [self.sigma_mag, None, self.sigma_rem, None,
                     self.dz_prism/10., self.width_max, self.max_amp*100.],
                    "Inversion parameters")
            if not okButton:
                print("\nNo inversion done: no inversion parameters given")
                return False
            self.sigma_mag = float(results[0])
            self.sigma_rem = float(results[2])
            self.depth_ref = float(results[4])
            self.width_max = int(results[5])
            self.max_amp = float(results[6])/100.
        else:
            results, okButton = dialog(
                    ["sigma_grav",
                     "\n",
                     "sigma_rho",
                     "\n",
                     "depth_ref [m]",
                     "width_max [samples]",
                     "max_amp [%]"],
                    ["e", "l", "e", "l", "e", "e", "e"],
                    [self.sigma_grav, None, self.sigma_rho, None,
                     self.dz_prism/10., self.width_max, self.max_amp*100.],
                    "Inversion parameters")
            if not okButton:
                print("\nNo inversion done: no inversion parameters given")
                return False
            self.sigma_grav = float(results[0])
            self.sigma_rho = float(results[2])
            self.depth_ref = float(results[4])
            self.width_max = int(results[5])
            self.max_amp = float(results[6])/100.
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
            fo.write("File(s):\n")
            # for i,fi in enumerate(data_files):
            #     fo.write(f"   {fi} as type {type_values[data_types[i]]}, "
            #              +f"measurement height {sensor_heights[i]}\n")
            fo.write(f"\nEarth's field: strength: {self.earth.f} nT, "
                     + f"inclination: {self.earth.inc} deg, "
                     + f"declination: {self.earth.dec} deg\n")
            fo.write("\nParameters\n")
            if self.sus_inv:
                fo.write("   Invert for susceptibilities\n")
            if self.rem_inv:
                fo.write("   Invert for remanence\n")
            if self.rho_inv:
                fo.write("   Invert for densities\n")
            fo.write(f"   Maximum number of iterations: {self.max_iter}\n")
            fo.write(f"   Stop if relative misfit < {self.max_diff_fac*100}% "
                     + "or relative misfit variation < "
                     + f"{self.max_rel_diff*100}%\n")
            fo.write(f"   Lambda: Initial: {self.lam}, minimum: " +
                     f"{self.lam_min}, factor per iteration: {self.lam_fac}\n")
            fo.write(f"   Gamma:  Initial: {self.gam}, minimum: {self.gam_min}"
                     + f", factor per iteration: {self.gam_fac}\n")
            fo.write("\nInitial prism area\n")
            fo.write(f"   West:  {self.xprism_min}, East:  {self.xprism_max}, "
                     + f"initial size in X: {self.dx_prism}, "
                     + f"min_size_x: {self.min_size_x}\n")
            fo.write(f"   South: {self.yprism_min}, North: {self.yprism_max}, "
                     + f"initial size in Y: {self.dy_prism}, "
                     + f"min_size_y: {self.min_size_y}\n")
            fo.write(f"   Top:   {self.zprism_min}, Bottom: {self.zprism_max},"
                     + f" initial size in Z: {self.dz_prism}, "
                     + f"min_size_z: {self.min_size_z}\n")
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
            fo.write("   Reference depth for weighting of parameter "
                     + f"variability: {self.depth_ref}\n")
            fo.write("   Size of area around a point for local maximum "
                     + f"determination: {self.width_max} points\n")
            fo.write("   Minimum relative amplitude for prism split: "
                     + f"{self.max_amp*100}%\n")
