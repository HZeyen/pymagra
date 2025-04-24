# -*- coding: utf-8 -*-
"""
Last modified on Feb 25 2025

@author: Hermann Zeyen <hermann.zeyen@universite-paris-saclay.fr>
         Universite Paris-Saclay, France
"""

import numpy as np


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
        self.data_type = typ
        if "M" in typ.upper():
            self.GM = "M"
        else:
            self.GM = "G"
        self.galile = 6.674e-6
        # No idea why in the following magnetization calculation the Earth's field
        # must be divided by 1000. Comparison of results with analytical formula for a
        # sphere and with Plouff's program gave this factor.
        self.sus = sus / (4 * np.pi)
        self.rem = rem
        self.inc = np.radians(rem_i)
        self.dec = np.radians(rem_d)
        self.rho = rho
        self.rk = np.zeros(3)
        self.rim = np.zeros(3)
        if np.isclose(rem, 0.0):
            self.sus2rem(sus)
        # self.rem = rem
        # self.rem_i = rem_i
        # self.rem_d = rem_d
        self.rim = np.array([earth.cdci, earth.sdci, earth.sie])
        self.xfac = np.zeros(3)
        self.yfac = np.zeros(3)
        self.zfac = np.zeros(3)
        # self.ini_okb(rem, rem_d, rem_i, rho)
        # self.corps = np.zeros((5, 1))
        # self.corps[:, 0] = np.array([rem, rem_d, rem_i, rho, 12])
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
        self.x = x
        self.y = y
        self.z = z
        self.der_flag_mag = False
        self.der_flag_grav = False
        self.x_neigh = []
        self.y_neigh = []
        self.z_neigh = []
        self.typ = "O"

    def setsus(self, sus):
        """
        Define susceptibility of body and convert to magnetization

        Parameters
        ----------
        sus : float
            Susceptibility of body in SI units
        """
        self.sus = sus
        self.sus2rem(sus)

    def getsus(self):
        """
        Return susceptibility of body

        Returns
        -------
        sus : float
            Susceptibility of body in SI units
        """
        return self.sus

    def setrem(self, rem):
        self.rem = rem
        self.ini_okb(self.rem, self.rem_d, self.rem_i, self.rho)

    def sus2rem(self, sus):
        """
        Calculate remanent magnetization in A/m given susceptibility and
        parameters of Earth's magnetic field'
        """
        self.rem = sus * self.earth.f
        self.rem_i = self.earth.inc
        self.rem_d = self.earth.dec
        self.ini_okb(self.rem, self.rem_d, self.rem_i, self.rho)

    def grav_prism(self, xp, yp, zp):
        """
        Calculate gravity effect of a prism at one point

        Parameters
        ----------
        xp : float
            X position of measurement point (easting).
        yp : float
            Y position of measurement point (northing).
        zp : float
            Height of measurement point (positive downward).

        Returns
        -------
        float
            Gravity effect of body at point (xp, yp, zp).

        """
        return self.calc_ano(xp, yp, zp)

    def calc_ano(self, xp, yp, alti):
        """
        Calculation of gravity or magnetic effect of a body

        Parameters
        ----------
        xp : float
            X position of measurement point (easting).
        yp : float
            Y position of measurement point (northing).
        alti : float
            Height of measurement point (positive downward).

        Returns
        -------
        c : float
            Calculated effect of body at point (xp, yp, zp).

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

        np_fac, nfac = self.x_face.shape
        c = 0.0
        for ifac in range(nfac):
            self.xfac = self.x_face[:, ifac]
            self.yfac = self.y_face[:, ifac]
            self.zfac = self.z_face[:, ifac]

            zfa_pt = self.zfac - alti
            yfa_pt = self.yfac - yp
            xfa_pt = self.xfac - xp
            # ATTN au passage de coord. ( x <-> y )    ???
            anom = -self.Okabe1(yfa_pt, xfa_pt, zfa_pt)
            if not np.isfinite(anom):
                print(f"face {ifac}, point ({xp}, {yp}): {anom}")
                continue
            c += anom
        return c

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
        Calculation of magnetic anomaly produced by one face with "Nsom"
        sides within a common plane at origin (0,0,0).

        Parameters
        ----------
        x, y, z : numpy float 1D arrays
            coordinates of the face ordered counterclock-wise

        Returns
        -------
        Magnetic effect of the face
        """
        # FERMETURE DU POLIGONE
        if len(x) < 3:
            return 0.0
        cosp, sinp, coste, sinte = self.rotation3D(x, y, z)
        cmpeff = (
            self.rim[0] * sinp * coste + self.rim[1] * sinp * sinte + self.rim[2] * cosp
        )
        if np.isclose(cmpeff, 0.0):
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
