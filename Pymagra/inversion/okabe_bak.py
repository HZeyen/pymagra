# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 15:56:23 2025

@author: Hermann
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

dir0 = r"E:\Daten\Magnetics\test_mag_topo"
os.chdir(dir0)

app = QtWidgets.QApplication(sys.argv)


class Earth_mag:
    """

    Class stores the components of the Earth's magnetic field and the direction
    (co)sines

    """

    def __init__(self, intensity, inclination, declination):
        self.f = intensity
        self.inc = inclination
        self.dec = declination
        self.earth_components()

    def earth_components(self):
        self.cde = np.cos(np.radians(self.dec))
        self.sde = np.sin(np.radians(self.dec))
        self.cie = np.cos(np.radians(self.inc))
        self.sie = np.sin(np.radians(self.inc))
        self.cdci = self.cde * self.cie
        self.sdci = self.sde * self.cie
        self.eh = self.f * self.cie
        self.ex = self.f * self.cdci
        self.ey = self.f * self.sdci
        self.ez = self.f * self.sie
        return True


class Okabe:
    """
    This class allows calculating the gravitational and magnetic effects of
    one rectangular vertical prism with inclined upper and lower surfaces

    use : prism = Okabe(x, y, z, sus, rem, rem_i, rem_d, rho, earth, typ)
    x : numpy float vector of length 2
        minimum and maximum x-coordinate of prism (E-W direction, in meters)
    y : numpy float vector of length 2
        minimum and maximum y-coordinate of prism (N-S direction, in meters)
    z : numpy float vector of length 8
        Z coordinate [m] of every corner point in the following order:
            upper surface:
            z[0]: z[x[0], y[0]], z[1]: z[x[1], y[0]], z[2]: z[x[1], y[1]]
            z[3]: z[x[0], y[1]]
            lower surface:
            z[4]: z[x[0], y[0]], z[5]: z[x[1], y[0]], z[6]: z[x[1], y[1]]
            z[7]: z[x[0], y[1]]
        Z is positive upward ! I.e., topography is usually positive and body
        corners negative or at least smaller than topography.
    sus : float
        susceptibility of prism in SI units
    rem : float
        Remanent magnetization intensiy [A/m].
        If rem is zero, induced magnetization is supposed and rem is replaced
        by the susceptibility transformed in A/m
    rem_i : float
        Inclination angle of remanent magnetization [degrees]
    rem_d : float
        Declination angle of remanent magnetization [degrees]
    rho : float
        Density [kg/m3]
    earth : Instance of class Earth_mag
        See e.g. pymagra/in_out/earth.py
    typ : str, optional; default: "M"
        If "M", magnetic effect calculated; if "G": gravity effect calculated

    For the calculation, each surface is subdivided into two triangles with
    vertices in clockwise sence when looking at the surface from the outside.

    """

    def __init__(self, x, y, z, sus, rem, rem_i, rem_d, rho, earth, typ="M"):
        self.earth = earth
        self.GM = typ.upper()
        self.galile = 6.674e-6
        # No idea why in the following magnetization calculation the Earth's field
        # must be divided by 2 and not by my_0*10**9 (my0 = 4pi*10**-7 and 10**9
        # because of the Earth's field being fiven in nT). Comparison of results
        # with analytical formula for a sphere and with Plouff's program gave this
        # factor of 200*pi.
        if np.isclose(rem, 0.0):
            # rem = sus*earth.f/(400*np.pi)
            rem = sus * earth.f * 0.5
            rem_i = earth.inc
            rem_d = earth.dec
        self.rk = np.zeros(3)
        self.rim = np.zeros(3)
        self.rim = np.array([earth.cdci, earth.sdci, earth.sie])
        self.xfac = np.zeros(3)
        self.yfac = np.zeros(3)
        self.zfac = np.zeros(3)
        self.ini_okb(rem, rem_d, rem_i, rho)
        self.corps = np.zeros((5, 1))
        self.corps[:, 0] = np.array([rem, rem_d, rem_i, rho, 12])
        self.ncorps = 1
        self.x_face = np.zeros((4, 12))
        self.y_face = np.zeros((4, 12))
        self.z_face = np.zeros((4, 12))
        self.x_face[:, 0] = np.array([x[0], x[1], x[1], x[0]])
        self.y_face[:, 0] = np.array([y[0], y[0], y[1], y[0]])
        self.z_face[:, 0] = np.array([z[0], z[1], z[2], z[0]])
        self.x_face[:, 1] = np.array([x[0], x[1], x[0], x[0]])
        self.y_face[:, 1] = np.array([y[0], y[1], y[1], y[0]])
        self.z_face[:, 1] = np.array([z[0], z[2], z[3], z[0]])
        self.x_face[:, 2] = np.array([x[0], x[1], x[1], x[0]])
        self.y_face[:, 2] = np.array([y[0], y[1], y[0], y[0]])
        self.z_face[:, 2] = np.array([z[4], z[6], z[5], z[4]])
        self.x_face[:, 3] = np.array([x[0], x[0], x[1], x[0]])
        self.y_face[:, 3] = np.array([y[0], y[1], y[1], y[0]])
        self.z_face[:, 3] = np.array([z[4], z[7], z[6], z[4]])
        self.x_face[:, 4] = np.array([x[0], x[1], x[1], x[0]])
        self.y_face[:, 4] = np.array([y[0], y[0], y[0], y[0]])
        self.z_face[:, 4] = np.array([z[4], z[5], z[1], z[4]])
        self.x_face[:, 5] = np.array([x[0], x[1], x[0], x[0]])
        self.y_face[:, 5] = np.array([y[0], y[0], y[0], y[0]])
        self.z_face[:, 5] = np.array([z[4], z[1], z[0], z[4]])
        self.x_face[:, 6] = np.array([x[0], x[1], x[1], x[0]])
        self.y_face[:, 6] = np.array([y[1], y[1], y[1], y[1]])
        self.z_face[:, 6] = np.array([z[7], z[2], z[6], z[7]])
        self.x_face[:, 7] = np.array([x[0], x[0], x[1], x[0]])
        self.y_face[:, 7] = np.array([y[1], y[1], y[1], y[1]])
        self.z_face[:, 7] = np.array([z[7], z[3], z[2], z[7]])
        self.x_face[:, 8] = np.array([x[1], x[1], x[1], x[1]])
        self.y_face[:, 8] = np.array([y[0], y[1], y[1], y[0]])
        self.z_face[:, 8] = np.array([z[5], z[6], z[2], z[5]])
        self.x_face[:, 9] = np.array([x[1], x[1], x[1], x[1]])
        self.y_face[:, 9] = np.array([y[0], y[1], y[0], y[0]])
        self.z_face[:, 9] = np.array([z[5], z[2], z[1], z[5]])
        self.x_face[:, 10] = np.array([x[0], x[0], x[0], x[0]])
        self.y_face[:, 10] = np.array([y[0], y[1], y[1], y[0]])
        self.z_face[:, 10] = np.array([z[4], z[3], z[7], z[4]])
        self.x_face[:, 11] = np.array([x[0], x[0], x[0], x[0]])
        self.y_face[:, 11] = np.array([y[0], y[0], y[1], y[0]])
        self.z_face[:, 11] = np.array([z[4], z[0], z[3], z[4]])

    def calc_ano(self, carte, x1, x2, dx, y1, y2, dy, alti):
        """
        convention du rangement dans la matrice "carte"

               carte(0, nrow-1)              carte(nrow-1,ncol-1)
                 y2!--------------------------!
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                   !                          !
                 y1!--------------------------!
                   x1                        x2
               carte(0,0)                carte(0, ncol-1)

        _____________________________________________________________________
        |      coord entry                          coord OKABE        |
        |                                                                   |
        |    | Z                                       _________________    |
        |    |           ________                    /|         X (Nord)    |
        |    |          /       /|    Il faut       / |                     |
        |    |         /       / |    passer       /  |       ________      |
        |    |        /_______/  |    d'un        /   |      /       /|     |
        |    |    Y/  |       |  |    systeme  Y /    |     /       / |     |
        |    |    /   |       | /       a      (Est)  |    /_______/  |     |
        |    |   /    |       |/      l'autre         |    |       |  |     |
        |    |  /     |_______/                       |    |       | /      |
        |    | /                                    Z |    |       |/       |
        |    |/_______________                        V    |_______/        |
        |                      X                                            |
        |___________________________________________________________________|
        """

        nrow, ncol = carte.shape
        carte[:, :] = 0.0
        np_fac, nfac = self.x_face.shape
        for ifac in range(nfac):
            self.xfac = self.x_face[:, ifac]
            self.yfac = self.y_face[:, ifac]
            self.zfac = self.z_face[:, ifac]

            zfa_pt = alti - self.zfac
            for ir in range(nrow):
                yp = y1 + ir * dy
                yfa_pt = self.yfac - yp
                for ic in range(ncol):
                    xp = x1 + ic * dx
                    xfa_pt = self.xfac - xp
                    # ATTN au passage de coord. ( x <-> y )    ???
                    anom = -self.Okabe1(
                        np.copy(yfa_pt), np.copy(xfa_pt), np.copy(zfa_pt)
                    )
                    if not np.isfinite(anom):
                        print(f"face {ifac}, row {ir}, col {ic}: {anom}")
                        sys.exit()
                    carte[ir, ic] = carte[ir, ic] + anom
        return carte

    def ini_okb(self, aim, deca, dipa, rho):
        if self.GM == "M":
            ad = deca * np.pi / 180.0
            ai = dipa * np.pi / 180.0
            self.rk[0] = aim * np.cos(ai) * np.cos(ad)
            self.rk[1] = aim * np.cos(ai) * np.sin(ad)
            self.rk[2] = aim * np.sin(ai)
            self.contra = 0.0
        else:
            self.contra = self.galile * rho
            # No idea whay, but the results for gravity calculation must be divided by 2
            # to get the correct result (test e.g. with Bouguer plate, but also comparison
            # with Nagy program)
            self.contra *= 0.5
            self.rim[0] = 0.0
            self.rim[1] = 0.0
            self.rim[2] = 1.0
            self.rk[0] = 0.0
            self.rk[1] = 0.0
            self.rk[2] = 1.0
            self.deb = True

    def rotation3D(self, x, y, z):
        xi = x[0]
        yi = y[0]
        zi = z[0]
        xj = x[1]
        yj = y[1]
        zj = z[1]
        xk = x[2]
        yk = y[2]
        zk = z[2]
        sxy = xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj)
        syz = yi * (zj - zk) + yj * (zk - zi) + yk * (zi - zj)
        szx = zi * (xj - xk) + zj * (xk - xi) + zk * (xi - xj)
        rr = syz * syz + szx * szx
        r3d = np.sqrt(rr + sxy * sxy)
        cosp = -sxy / r3d
        r = np.sqrt(rr)
        sinp = r / r3d
        if np.isclose(rr, 0.0):
            coste = 1.0
            sinte = 0.0
        else:
            coste = -syz / r
            sinte = -szx / r
        return cosp, sinp, coste, sinte

    def rotation2D(self, x, y, z, cosp, sinp, coste, sinte):
        a = coste * x + sinte * y
        xm = a * cosp - z * sinp
        ym = y * coste - x * sinte
        zm = a * sinp + z * cosp
        return xm, ym, zm

    def Okabe1(self, x, y, z):
        """
         ________________________________________________________________
        |                                                               |
        |  Reference : Okabe, M., Analytical expressions for gravity    |
        |     anomalies due to polyhedral bodies and translation into   |
        |     magnetic anomalies, Geophysics, 44, (1979), p 730-741.    |
        |_______________________________________________________________|
            Character*1 GM
            Parameter (Maxsom = 51)
            Dimension X(Nsom),Y(Nsom),Z(Nsom)
            Dimension Xx(Maxsom),Yy(Maxsom),Zz(Maxsom)
            Common/Cosdr/Rk(3),Rim(3),Aim,Contra,GM
        """
        if self.GM == "M":
            okabe1 = self.Okbmag(x, y, z)
        else:
            okabe1 = self.Okbgra(x, y, z)
        return okabe1

    def Okbgra(self, x, y, z):
        """
          ATTN: l'entree (x, y, z) est detruite
          ---------------------------------------------------------------------
           Calcul de l'anomalie gravimetrique creee par une facette de 3
           sommets COPLANAIRES au point origine (0,0,0).
          ---------------------------------------------------------------------
        # Real*8 Deps,Resul
        # Dimension X(51),Y(51),Z(51)
        # Common /Keps/Deps,Eps
        """
        # ! FERMETURE DU POLIGONE
        if len(x) < 3:
            return 0.0
        cosp, sinp, coste, sinte = self.rotation3D(x, y, z)
        if np.isclose(abs(cosp), 0.0):
            return 0.0
        # Rotations de 'teta' et 'phi' (formule (17) de Okabe)
        xt, yt, zt = self.rotation2D(x, y, z, cosp, sinp, coste, sinte)

        resul = 0.0
        for i in range(3):
            x1 = xt[i]
            x2 = xt[i + 1]
            dx = x2 - x1
            y1 = yt[i]
            y2 = yt[i + 1]
            dy = y2 - y1
            r = np.sqrt(dx * dx + dy * dy)
            if np.isclose(r, 0.0):
                continue
            cosps = dx / r
            sinps = dy / r
            z2 = zt[i + 1]
            z1 = zt[i]
            res = self.Okg(cosps, sinps, x2, y2, z2) - self.Okg(
                cosps, sinps, x1, y1, z1
            )
            resul += res
        # return resul*cosp
        return resul * self.contra * cosp

    def Okg(self, c, s, x, y, z):
        t = 0.0
        r = np.sqrt(x * x + y * y + z * z)
        if r > 0.0:
            if abs(z) > 0.0 and abs(c) > 0.0:
                t = (x * c + (1 + s) * (y + r)) / (z * c)
                t = -2.0 * z * np.arctan(t)
            rprim = x * c + y * s + r
            if rprim > 0.0:
                t += (x * s - y * c) * np.log(rprim)
        return t

    def Okbmag(self, x, y, z):
        """
          ATTN: l'entree (X,Y,Z) est detruite
          ---------------------------------------------------------------------
           Calcul de l'anomalie magnetrique creee par une facette de "Nsom"
           sommets COPLANAIRES au point origine (0,0,0).
          ---------------------------------------------------------------------
        DIMENSION X(51),Y(51),Z(51)
        Real*8 Resul,Deps
        Character*1 GM
        Common/Cosdr/Rk(3),Rim(3),Aim,Contra,GM
        Common /Keps/ Deps,Eps
        """
        # FERMETURE DU POLIGONE
        if len(x) < 3:
            return 0.0
        cosp, sinp, coste, sinte = self.rotation3D(x, y, z)
        cmpeff = (
            self.rim[0] * sinp * coste + self.rim[1] * sinp * sinte + self.rim[2] * cosp
        )
        if abs(cmpeff) <= 0.0:
            return 0.0
        # Rotation des axes magnetiques
        xt = self.rk[0]
        yt = self.rk[1]
        zt = self.rk[2]
        at = coste * xt + sinte * yt
        xm = at * cosp - zt * sinp
        ym = yt * coste - xt * sinte
        zm = at * sinp + zt * cosp
        # Rotations de 'teta' et 'phi' (formule (17) de Okabe)
        xt, yt, zt = self.rotation2D(x, y, z, cosp, sinp, coste, sinte)

        resul = 0.0
        for i in range(3):
            xi = xt[i]
            xi1 = xt[i + 1]
            dx = xi1 - xi
            yi = yt[i]
            yi1 = yt[i + 1]
            dy = yi1 - yi
            rr = dx * dx + dy * dy
            r = np.sqrt(rr)
            if np.isclose(r, 0.0):
                res = 0.0
            else:
                cosps = dx / r
                sinps = dy / r
                # Rotation dans le plan de la facette (formule 20)
                # A partir d'ici  X -> Qsi  et Y -> Eta
                x2 = yi1 * sinps + xi1 * cosps
                y2 = yi1 * cosps - xi1 * sinps
                x1 = yi * sinps + xi * cosps
                y1 = yi * cosps - xi * sinps
                z2 = zt[i + 1]
                z1 = zt[i]
                res = self.Okm(xm, ym, zm, cosps, sinps, x2, y2, z2)
                res -= self.Okm(xm, ym, zm, cosps, sinps, x1, y1, z1)
            resul += res
        return resul * cmpeff

    def Okm(self, xm, ym, zm, c, s, x, y, z):
        #   Real*8 Deps
        # Common /Keps/ Deps,Eps
        aa = y * y + z * z
        r = np.sqrt(x * x + aa)
        if np.isclose(abs(r), 0.0):
            return 0.0
        if abs(z) > 0.0 and abs(c) > 0.0:
            t = zm * np.arctan((aa * (s / c) - x * y) / (z * r))
        else:
            t = 0.0
        rprim = x + r
        if rprim <= 0.0:
            t += (xm * s - ym * c) * np.log(r - x)
        else:
            t += (ym * c - xm * s) * np.log(rprim)
        return t


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
                "Synthetic model file has only {ncol} columns:\n"
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


def store_gxf(file, data, x0, y0, dx, dy):
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
                if nc == 8 or ix == cols - 1:
                    fo.write("\n")
                    nc = 0


earth = Earth_mag(50000.0, 90.0, 0.0)
xp, yp, zz, sus, rem, rem_i, rem_d, rho = read_synthetic_model()
zp = np.zeros(8)
zp[:2] = -zz[0][0]
zp[2:4] = zp[:2] + 0.0
zp[4:] = -zz[0][1]
# typ = "M"
typ = "G"
prism = Okabe(
    xp[0], yp[0], zp, sus[0], rem[0], rem_i[0], rem_d[0], rho[0], earth, typ=typ
)
x0 = 0.0
x1 = 15.0
dx = 0.5
y0 = 0.0
y1 = 15.0
dy = 0.5
x = np.arange(x0, x1 + dx / 2.0, dx)
y = np.arange(y0, y1 + dy / 2.0, dy)
nx = len(x)
ny = len(y)
calc = np.zeros((nx, ny))
calc += prism.calc_ano(calc, x0, x1, dx, y0, y1, dy, 0.0)

fig, ax = plt.subplots(1, 1, figsize=(10, 15))
pl = ax.imshow(np.flip(calc, axis=0), cmap="rainbow", extent=[x0, x1, y0, y1])
ax.set_xlabel("Easting[m]")
ax.set_ylabel("Northing [m]")
ax.set_title("Synthetic model Okabe")
if typ == "M":
    store_gxf("Okabe_mag.gxf", calc, x0, y0, dx, dy)
else:
    store_gxf("Okabe_grav.gxf", calc, x0, y0, dx, dy)
