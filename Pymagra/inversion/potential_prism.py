# -*- coding: utf-8 -*-
"""
Last modified: Nov 22, 2024

@author: Hermann Zeyen
        University Paris-Saclay

Contains two classes:
    Prism: Defines position and properties of one prism
        Contains the following methods:
        - __init__
        - mag_prism (calculates magnetic effect at one point)
        - grav_prism (calculates gravity effect at one point)
        - change_props (modifies one or several properties)
        - change_coor (modifies one or several prism coordinates)

    Prism_calc: Allows defining a model composed of a series of prisms and
    calculation of effects at a vector of points.

    Contains the following methods:
        - __init__
        - add_prism: adds an entrance of class Prism to a dictionary
        - remove_prism: removes an entrance of classe Prism from
          dictionary
        - mag_forward: Solves magnetic forward problem for all prisms
          at all points
        - mag_deriv: Calculates derivatives of magnetic effect with
          respect to magnetic properties (susceptibility and/or
          remanence)
        - grav_forward: Solves gravity forward problem for all prisms
          at all points
        - grav_deriv: Calculates derivatives of magnetic effect with
          respect to density
        - create_Frechet: Assembles the derivative vectors of each
          prism into a Freceht matrix
        - get_max_prisms: Find all prisms located below the strongest
          absolute maximum of magnetic and gravity fields.
          Usually used during inversion procedure where
          the data correspond to actual misfit.
        - split: split a prism into up to 8 prisms having half size in
          all directions as long as prisms stay larger than
          a given minimum size.

"""

from . import mag_grav_utilities as utils
from ..in_out.earth import Earth_mag as Earth
import numpy as np


class Prism(Earth):
    """
    Class contains the coordinates and properties of a vertical prism with
    faces parallel to the coordinate system

    Uses collections of functions from mag_grav_utilities.py

    Input
    -----

    x: numpy float array with 2 values
        x[0]: coordinate of the western edge of the prism [m]
        x[1]: coordinate of the eastern edge of the prism [m]

    y: numpy float array with 2 values
        y[0]: coordinate of the southern edge of the prism [m]
        y[1]: coordinate of the northern edge of the prism [m]

    z: numpy float array with 2 values
        z[0]: coordinate of the upper edge of the prism [m]
        z[1]: coordinate of the lower edge of the prism [m]
        Positive downward

    sus: susceptibility [SI]

    rem: Remanent magnetization [A/m]

    inc: Inclination fo the remanent magnetization [degrees]

    dec: Declination fo the remanent magnetization [degrees]

    dens: Density of the prism [kg/m3] (not yet used, TODO)

    earth: class Earth_mag object
            contains the properties of the Earth's magnetic field

    Methods
    -------

    - __init__
    - mag_prism (calculates magnetic effect at one point)
    - grav_prism (calculates gravity effect at one point)
    - change_props (modifies one or several properties)
    - change_coor (modifies one or several prism coordinates)

    """

    def __init__(self, x, y, z, sus, rem, inc, dec, dens, earth):
        """
        x: numpy float 1D array of size 2
                coordinates of W and E faces
        y: numpy float 1D array of size 2
                coordinates of S and N faces
        z: numpy float 1D array of size 2
                coordinates of upper and lower faces (positive downward)
        sus: float
                susceptibility [SI]
        rem: float
                remanent magnetization [A/m]
        inc: float
                inclination of remanent magnetization [degrees]
        dec: float
                declination of remanent magnetization [degrees]
        dens: float
                density [kg/m3]
        earth: object of type Earth_mag found in file mag_grav_utilities.py
        """
# Susceptibility is transformed to cgs units which is used in function
# mag_prism
        self.G = 6.67e-6
        self.sus = sus/(4*np.pi)
        self.rem = rem
        self.inc = np.radians(inc)
        self.dec = np.radians(dec)
        self.rho = dens
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.z = np.copy(z)
        self.earth = earth
        self.comps()
        self.iga = np.ones(2, dtype=int)
        self.iga[0] = -1
        self.igb = np.ones(2, dtype=int)
        self.igb[0] = -1
        self.igh = np.ones(2, dtype=int)
        self.igh[0] = -1
        self.der_flag_mag = False
        self.der_flag_grav = False
        self.x_neigh = []
        self.y_neigh = []
        self.z_neigh = []

    def comps(self):
        """
        Calculate magnetization components
        """
        self.tx, self.ty, self.tz = utils.magnetization_components(
            self.sus, self.rem, self.dec, self.inc, self.earth)

    def mag_prism(self, xp, yp, zp):
        """
        Calculates the exact 3D magnetic effect of one vertical prism with
        its sides parallel to the coordinate system at one measurement point.
        Needs susceptibilities in cgs system (SI/4pi))

        Parameters
        ----------
        xp, yp, zp : 1D Numpy float arrays
            Coordinates of all calculation points

        Returns
        -------
        delx, dely, delz : float
            N-S, E-W and Z component of the effect of the body

        """
# a, b and h are the distances between prism edges and calculation point
        a = np.float64(self.y-yp)
        a[abs(a) < 1.e-10] = np.float64(0.)
        isa = np.sign(a)
        a2 = a**2

        b = np.float64(self.x-xp)
        b[abs(b) < 1.e-10] = np.float64(0.)
        isb = np.sign(b)
        b2 = b**2

        ija = np.mean(isa)
        ijb = np.mean(isb)

# Make sure inside h, the coordinates are stored smaller value first
# If body is located above the calculation point, absolute values of
# z coordiantes are stored, but the sign of the anomaly will be reversed
# Z prism is positive downwards, zp (z point is positive upward)
#
# TODO:
# What happens if calculation point is located within the body or at least
# between the two limits (beside the body)?
        # ht = np.float64(abs(self.z[0]-zp))
        # hb = np.float64(abs(self.z[1]-zp))
        ht = np.float64(abs(self.z[0]+zp))
        hb = np.float64(abs(self.z[1]+zp))
        if hb < ht:
            sgh = -1.
            h = np.array([hb, ht])
        elif hb > ht:
            sgh = 1.
            h = np.array([ht, hb])
# If body thickness is zero, no need to calculate
        else:
            return 0., 0., 0.
        if h[0] < 1.e-10:
            h[0] = 0.
        h2 = h**2

        g1 = 1.
        g2 = 1.
        g3 = 1.
        t1 = 0.
        t2 = 0.
        w1 = 0.
        w2 = 0.
# Loop over all edges
        for k in range(2):
            hk = h[k]
            for i in range(2):
                ai = a[i]
                isai = isa[i]
                ad = abs(ai)
                igna = self.iga[i]
                aa = a2[i]
                for j in range(2):
                    bj = b[j]
                    isbj = isb[j]
                    ab = ai*bj
                    bb = b2[j]
                    aabb = aa+bb
                    igab = igna*self.igb[j]
                    bd = abs(bj)
                    isgn = igab*self.igh[k]
                    ft1 = 0.
                    ft2 = 0.
                    rr = aabb+h2[k]
                    if np.isclose(rr, 0.):
                        return 0., 0., 0.
                    r = np.sqrt(rr)
                    fg1 = bj+r
                    fg2 = hk+r
                    fg3 = ai+r
#
#  Needed for interior/edge points and for negative values of ai/bj that lead
#  to an undefined log (log of zero or negative value)
#
                    if np.isclose(hk, 0.) &\
                            (isbj == 0 or isai != 0 or ijb < 1):
                        if isbj == 0:
                            if ija < 0:
                                fg3 = 1./ad
                            elif ija == 0:
                                return 0., 0., 0.
                        elif isai == 0:
                            if ijb < 0:
                                fg1 = 1./bd
                            elif ijb == 0:
                                return 0., 0., 0.
                    elif abs(ab) > 0.:
                        ft1 = bj*hk/(ai*r)
                        ft2 = ai*hk/(bj*r)
#  Negative arctan and reciprocal log terms
                    if isgn != 1:
                        g1 /= fg1
                        g2 /= fg2
                        g3 /= fg3
                        ft1 = -ft1
                        ft2 = -ft2
                    else:
                        g1 *= fg1
                        g2 *= fg2
                        g3 *= fg3
                    if abs(ab) > 0.:
                        w1, t1 = utils.tandet(ft1, w1, t1)
                        w2, t2 = utils.tandet(ft2, w2, t2)
        g1 = np.log(g1)
        g2 = np.log(g2)
        g3 = np.log(g3)
        t1 = -(np.arctan(w1)+t1)
        t2 = -(np.arctan(w2)+t2)
        t3 = -(t1+t2)
        dex = (self.tx*t1+self.ty*g2+self.tz*g1)*sgh
        dey = (self.tx*g2+self.ty*t2+self.tz*g3)*sgh
        dez = (self.tx*g1+self.ty*g3+self.tz*t3)*sgh
        return dex, dey, dez

    def grav_prism(self, xp, yp, zp):
        """
        Calculates the exact 3D gravity effect of a vertical rectangular prism.

        Parameters
        ----------
        xp, yp, zp : float
            Coordinates of the point where the effect is to be calculated

        Returns
        -------
        g : float
            Gravity effect of the body

        """
        if np.isclose(self.z[0], self.z[1]):
            return 0.
        x = np.float64(self.x-xp)
        y = np.float64(self.y-yp)
        z = np.float64(self.z-zp)
        x[abs(x) < 1.e-5] = np.float64(1.e-5)
        y[abs(y) < 1.e-5] = np.float64(1.e-5)
        z[abs(z) < 1.e-5] = np.float64(1.e-5)
        x2 = x**2
        y2 = y**2
        z2 = z**2
        log_x = np.ones(2)
        log_y = np.ones(2)
        atan_z = np.zeros(2)
        sum_pi = np.zeros(2)
        sig = [1, -1]
        for k in range(2):
            for j in range(2):
                rr = z2[k]+y2[j]
                sgn = sig[j]*sig[k]
                for i in range(2):
                    s = sgn*sig[i]
                    r2 = rr+x2[i]
                    r = np.sqrt(r2)
                    if s > 0:
                        if not np.isclose(-y[j], r):
                            log_x[i] *= y[j]+r
                        if not np.isclose(-x[i], r):
                            log_y[j] *= x[i]+r
                    else:
                        if not np.isclose(-y[j], r):
                            log_x[i] /= y[j]+r
                        if not np.isclose(-x[i], r):
                            log_y[j] /= x[i]+r
                    fac = (x[i]+y[j]+r)/z[k]*s
                    atan_z[k], sum_pi[k] = utils.tandet(fac, atan_z[k],
                                                        sum_pi[k])
        g = np.sum(x*np.log(log_x) + y*np.log(log_y) +
                   2.*z*(np.arctan(atan_z)+sum_pi))
        return g*self.rho*self.G

    # def okabe(self):
        

    def change_props(self, sus=None, rem=None, inc=None, dec=None, dens=None):
        """
        Change properties of a prism for each entry that is not None

        Parameters
        ----------
        sus : float, optional; default: None
            Susceptibility [SI system].
        rem : float, optional; default: None
            Remanent magnetization [A/m].
        inc : float, optional; default: None
            Inclination of remanent magnetization [degrees].
        dec : float, optional; default: None
            Declination of remanent magnetization [degrees].
        dens : float, optional; default: None
            Density [kg/m3].

        Returns
        -------
        None.

        """
        if sus is not None:
            self.sus = sus/(4*np.pi)
        if rem is not None:
            self.rem = rem
        if inc is not None:
            self.inc = np.radians(inc)
        if dec is not None:
            self.dec = np.radians(dec)
        if dens is not None:
            self.rho = dens
        self.tx, self.ty, self.tz = utils.magnetization_components(
            sus, rem, dec, inc, self.earth)

    def change_coor(self, xmin=None, xmax=None, ymin=None, ymax=None,
                    zmin=None, zmax=None):
        """
        Change coordinates of prism faces for all entries that are not None.

        Parameters
        ----------
        xmin : float, optional; default: None
            Coordiante of western face of prism [m].
        xmax : float, optional; default: None
            Coordiante of eastern face of prism [m].
        ymin : float, optional; default: None
            Coordiante of southern face of prism [m].
        ymax : float, optional; default: None
            Coordiante of northern face of prism [m].
        zmin : float, optional; default: None
            Coordiante of upper face of prism ([m] positive below surface).
        zmax : float, optional; default: None
            Coordiante of lower face of prism ([m] positive below surface).

        Returns
        -------
        None.

        """
        if xmin is not None:
            self.x[0] = xmin
        if xmax is not None:
            self.x[1] = xmax
        if ymin is not None:
            self.y[0] = ymin
        if ymax is not None:
            self.y[1] = ymax
        if zmin is not None:
            self.z[0] = zmin
        if zmax is not None:
            self.z[1] = zmax


class Prism_calc(Prism, Earth):
    """
    Contains the following methods:

    - __init__
    - add_prism: adds an entrance of class Prism to a dictionary
    - remove_prism: removes an entrance of classe Prism from dictionary
    - mag_forward: Solves magnetic forward problem for all prisms at all points
    - mag_deriv: Calculates derivatives of magnetic effect with respect to
      magnetic properties (susceptibility and/or remanence)
    - grav_forward: Solves gravity forward problem for all prisms at all points
    - grav_deriv: Calculates derivatives of magnetic effect with respect to
      density
    - create_Frechet: Assembles the derivative vectors of each prism into a
      Frechet matrix
    - get_max_prisms: Find all prisms located below the strongest absolute
      maximum of magnetic and gravity fields. Usually used during inversion
      procedure where the data correspond to actual misfit.
    - split: split a prism into up to 8 prisms having half size in all
      directions as long as prisms stay larger than a given minimum size.

    """

    def __init__(self, e, min_size_x=0., min_size_y=0., min_size_z=0.):
        """
        Initialize dictionary of prisms used for magnetic and gravity
        calculation

        Parameters
        ----------
        e: Obejct of class Earth_mag (found in file mag_grav_utilities.py)
        min_size_x : float
            Minimum allowed size of prisms in X direction (E-W).
        min_size_y : float
            Minimum allowed size of prisms in Y direction (N-S).
        min_size_z : float
            Minimum allowed size of prisms in Z direction.

        Returns
        -------
        None.

        """
# self.G is the universal gravity constant multiplied by 10**5 so that results
#       are in mGal
        self.G = 6.67e-6
# Define dictionaray that will contain all model prisms
        self.prisms = {}
# self.n_prisms is the number of defined prisms
        self.n_prisms = 0
# self.n_max is the key of the last defined prism. Initially, n_max = n_prism.
#       However, if a prism other than the last defined one is removed from the
#       dictionary, it's key is erased, n_prism is reduced by one, but the
#       next prism that will be added to the dictionary will have key n_max+1.
#       In this case, n_prisms and n_max will be different
        self.n_max = -1
        self.min_size_x = min_size_x
        self.min_size_y = min_size_y
        self.min_size_z = min_size_z
        self.earth = e

    def add_prism(self, xpr, ypr, zpr, sus, rem, rinc, rdec, rho):
        """
        Add a prism with its properties to the dictionary.
        The key of this new prism will be self.n_max+1

        For the explanation of input parameters see Prism.__init__
        """
        self.n_max += 1
        self.prisms[self.n_max] = Prism(xpr, ypr, zpr, sus, rem, rinc, rdec,
                                        rho, self.earth)
        self.n_prisms += 1
        if self.n_prisms == 1:
            return
        x1 = xpr.min()
        x2 = xpr.max()
        y1 = ypr.min()
        y2 = ypr.max()
        z1 = zpr.min()
        z2 = zpr.max()
        keys = list(self.prisms.keys())
        for key in keys[:-1]:
            x3 = self.prisms[key].x.min()
            x4 = self.prisms[key].x.max()
            y3 = self.prisms[key].y.min()
            y4 = self.prisms[key].y.max()
            z3 = self.prisms[key].z.min()
            z4 = self.prisms[key].z.max()
            if (x3 == x2 or x4 == x1) and y2 > y3 and y1 < y4 and z2 > z3 and\
                    z1 < z4:
                self.prisms[self.n_max].x_neigh.append(key)
            if (y3 == y2 or y4 == y1) and x2 > x3 and x1 < x4 and z2 > z3 and\
                    z1 < z4:
                self.prisms[self.n_max].y_neigh.append(key)
            if (z3 == z2 or z4 == z1) and x2 > x3 and x1 < x4 and y2 > y3 and\
                    y1 < y4:
                self.prisms[self.n_max].z_neigh.append(key)

    def remove_prism(self, key):
        """
        Remove a prism for the dictionary
        n_prisms will be reduced by one, but not n_max.
        """
        try:
            del self.prisms[key]
            self.n_prisms -= 1
            for k in self.prisms.keys():
                if key in self.prisms[k].x_neigh:
                    del self.prisms[k].x_neigh[np.where(
                        np.array(self.prisms[k].x_neigh) == key)[0][0]]
                if key in self.prisms[k].y_neigh:
                    del self.prisms[k].y_neigh[np.where(
                        np.array(self.prisms[k].y_neigh) == key)[0][0]]
                if key in self.prisms[k].z_neigh:
                    del self.prisms[k].z_neigh[np.where(
                        np.array(self.prisms[k].z_neigh) == key)[0][0]]
            return True
        except KeyError:
            print(f"\nWarning in 'remove_prism': Prism {key} does not exist "
                  + "in dictionary.")
            return False

    def mag_forward(self, xp, yp, zp):
        """
        Calculate summed effect of all prisms on all field points

        Parameters
        ----------
        prisms : dictionary with Prism class objects
            Positions and properties of all prisms
        xp : numpy 1D float array [n_points]
            X-coordiante of field points
            [nr: number of rows in NS direction, nc: number columns in EW
            direction]
        yp : numpy 1D float array [n_points]
            Y-coordiante of field points
        zp : numpy 1D float array [n_points]
            Z-coordiante of field points

        Returns
        -------
        v : numpy 2D float array [n_points,5]
            Calculated anomalies.
            v[:,0] : X-component (N-S)
            v[:,1] : Y-component (E-W)
            v[:,2] : Z-component
            v[:,3] : horizontal component
            v[:,4] : total field component
        """
        n_points = len(xp)
        self.v = np.zeros((n_points, 5))
        for i in range(n_points):
            delx = 0.
            dely = 0.
            delz = 0.
            for key, val in self.prisms.items():
                dex, dey, dez = val.mag_prism(xp[i], yp[i], zp[i])
                delx += dex
                dely += dey
                delz += dez
            delh, delt = utils.compon(delx, dely, delz, self.earth)
            self.v[i, 0] += delx
            self.v[i, 1] += dely
            self.v[i, 2] += delz
            self.v[i, 3] += delh
            self.v[i, 4] += delt
            if np.mod(i+1, 100) == 0:
                print(f"Point {i+1}: Magnetic field calculated")
        return np.copy(self.v)

    def mag_deriv(self, xp, yp, zp, sus_inv=True, rem_inv=False):
        """
        Calculate summed effect of all prisms on all field points

        Parameters
        ----------
        prisms : dictionary with Prism class objects
            Positions and properties of all prisms
        xp : numpy 1D float array [n_points]
            X-coordiante of field points
            [nr: number of rows in NS direction, nc: number columns in EW
            direction]
        yp : numpy 1D float array [n_points]
            Y-coordiante of field points
        zp : numpy 1D float array [n_points]
            Z-coordiante of field points
        sus_inv : bool, optional
            If True susceptibility derivatives are calculated; Default: True
        rem_inv : bool, optional
            If True remanence derivatives are calculated; Default: False

        """
        n_points = len(xp)
        for key, val in self.prisms.items():
            if self.prisms[key].der_flag_mag:
                continue
            val.sus_der = np.zeros(n_points)
            val.rem_der = np.zeros(n_points)
            rem = val.rem
            sus = val.sus
            if sus_inv:
                val.rem = 0.
                val.sus = 1.
                val.comps()
                for i in range(n_points):
                    dex, dey, dez = val.mag_prism(xp[i], yp[i], zp[i])
                    deh, det = utils.compon(dex, dey, dez, self.earth)
                    self.prisms[key].sus_der[i] = det
                val.rem = rem
                val.sus = sus
                self.prisms[key].comps()
                self.prisms[key].der_flag_mag = True
            if rem_inv:
                val.sus = 0.
                val.rem = 1.
                val.comps()
                for i in range(n_points):
                    dexr, deyr, dezr = val.mag_prism(xp[i], yp[i], zp[i])
                    deh, det = utils.compon(dex, dey, dez, self.earth)
                    self.prisms[key].rem_der[i] = det
                val.sus = sus
                val.rem = rem
                self.prisms[key].comps()
                self.prisms[key].der_flag_mag = True
            print(f"Prism {key+1}: Magnetic derivative calculated")
        return True

    def grav_forward(self, xp, yp, zp):
        """
        Calculate summed gravity effect of all prisms on all field points

        Parameters
        ----------
        prisms : dictionary with Prism class objects
            Positions and properties of all prisms
        xp : numpy 1D float array [n_points]
            X-coordiante of field points
            [nr: number of rows in NS direction, nc: number columns in EW
            direction]
        yp : numpy 1D float array [n_points]
            Y-coordiante of field points
        zp : numpy 1D float array [n_points]
            Z-coordiante of field points

        Returns
        -------
        g : numpy 1D float array [n_points]
            Calculated anomalies.
        """
    # G is the universal gravity constant
    # in order to pass from m/s2 to mGal, it is multiplied by 10**5
        n_points = len(xp)
        self.g = np.zeros(n_points)
        for i in range(n_points):
            for key, val in self.prisms.items():
                self.g[i] += val.grav_prism(xp[i], yp[i], zp[i])
            if np.mod(i+1, 100) == 0:
                print(f"Point {i+1}: Gravi calculated")
        return np.copy(self.g)

    def grav_deriv(self, xp, yp, zp, deriv=True):
        """
        Calculate summed gravity effect of all prisms on all field points

        Parameters
        ----------
        prisms : dictionary with Prism class objects
            Positions and properties of all prisms
        xp : numpy 1D float array [n_points]
            X-coordiante of field points
            [nr: number of rows in NS direction, nc: number columns in EW
            direction]
        yp : numpy 1D float array [n_points]
            Y-coordiante of field points
        zp : numpy 1D float array [n_points]
            Z-coordiante of field points
        """
        n_points = len(xp)
        for key, val in self.prisms.items():
            if val.der_flag_grav:
                continue
            val.rho_der = np.zeros(n_points)
            rho = val.rho
            val.rho = 1.
            for i in range(n_points):
                self.prisms[key].rho_der[i] = \
                    val.grav_prism(xp[i], yp[i], zp[i])
            val.rho = rho
            self.prisms[key].der_flag_grav = True
            print(f"Prism {key+1}: Gravi derivatives calculated")
        return True

    def get_n_param(self, sus_inv, rem_inv, rho_inv):
        """
        Calculate total number of model parameters

        Parameters
        ----------
        sus_inv : bool
            If True, susceptibilities are model parameters
        rem_inv : bool
            If True, remanences are model parameters
        rho_inv : bool
            If True, densities are model parameters

        Return
        ------
            self.n_param : int
                Total number of parameters
        """
        self.n_param = 0
        if sus_inv:
            self.n_param += self.n_prisms
        if rem_inv:
            self.n_param += self.n_prisms
        if rho_inv:
            self.n_param += self.n_prisms
# add one more parameter corresponding to the unknown data zero level
        self.n_param += 1
        return self.n_param

    def create_Frechet(self, sus_inv, rem_inv, rho_inv, xp, yp, zp):
        """
        Calculate Freceht matrix

        Parameters
        ----------
        sus_inv : bool
            If True, susceptibilities are model parameters
        rem_inv : bool
            If True, remanences are model parameters
        rho_inv : bool
            If True, densities are model parameters
        xpm : 1D numpy float array (lenght: number of magnetic data points)
            X coordinates of magnetic data points
        ypm : 1D numpy float array (lenght: number of magnetic data points)
            Y coordinates of magnetic data points
        zpm : 1D numpy float array (lenght: number of magnetic data points)
            Z coordinates of magnetic data points
        xpg : 1D numpy float array (lenght: number of gravity data points)
            X coordinates of gravity data points
        ypg : 1D numpy float array (lenght: number of gravity data points)
            Y coordinates of gravity data points
        zpg : 1D numpy float array (lenght: number of gravity data points)
            Z coordinates of gravity data points

        Return
        ------
            self.Frechet : 2D numpy float matrix with size [self.n_dat,
            self.n_param]
                Frechet matrix
        """
        self.ndat = 0
        if sus_inv or rem_inv:
            n_data = len(xp)
            self.ndat += n_data
            self.mag_deriv(xp, yp, zp)
        if rho_inv:
            n_data = len(xp)
            self.ndat += n_data
            self.grav_deriv(xp, yp, zp)
        self.n_param = self.get_n_param(sus_inv, rem_inv, rho_inv)
        self.Frechet = np.zeros((self.ndat, self.n_param))
        icol = 0
        nrow = 0
        self.params = []
        if sus_inv:
            nrow1 = nrow + n_data
            for i, key in enumerate(self.prisms.keys()):
                self.Frechet[nrow:nrow1, i+icol] = self.prisms[key].sus_der
                self.params.append(self.prisms[key].sus)
            if not rem_inv:
                nrow += n_data
                nrow1 = nrow + n_data
            icol += self.n_prisms
        if rem_inv:
            for i, key in enumerate(self.prisms.keys()):
                self.Frechet[nrow:nrow1, i+icol] = self.prisms[key].rem_der
                self.params.append(self.prisms[key].rem)
            nrow += n_data
            icol += self.n_prisms
        if rho_inv:
            nrow1 = nrow + n_data
            for i, key in enumerate(self.prisms.keys()):
                self.Frechet[nrow:nrow1, i+icol] = self.prisms[key].rho_der
                self.params.append(self.prisms[key].rho)
        self.params.append(0.)
        self.Frechet[:, -1] = 1.
        self.params = np.array(self.params)
        return np.copy(self.Frechet)

    def create_smooth(self, sus_inv, rem_inv, rho_inv, sigma_sus, sigma_rem,
                      sigma_rho, depth_ref):
        """
        Create smoothing matrix. It has size [n_params,n_params].
        On the diagonal is the number of neighbouring blocks (2 to 4). at the
        position [i,j] value is -1. i,j are numbers of neighbouring blocks (not
        block keys but position of the block in the dictionary)
        In the actual version, no smoothing in Z is done

        Parameters
        ----------
        sus_inv : bool
            If True, susceptibilities are model parameters
        rem_inv : bool
            If True, remanences are model parameters
        rho_inv : bool
            If True, densities are model parameters
        sigma_sus : float
            Variability of susceptibility [SI units]
        sigma_rem : float
            Variability of remances [A/m]
        sigma_rho : float
            Variability of densities [kg/m3]
        depth_ref : float
            Reference depth to increase smoothing with depth

        Returns
        -------
        S : 2D numpy float array
            Smoothing matrix mith size [n_params,n_params].

        """
        S = np.zeros((self.n_param, self.n_param))
        SS = np.zeros((self.n_prisms, self.n_prisms))
        z_fac = np.zeros(self.n_prisms)
        keys = np.array(list(self.prisms.keys()))
# Check whether neighbouring block in X direction exists
        for i, key in enumerate(keys):
            z_fac[i] = (self.prisms[key].z[0] + self.prisms[key].z[1])\
                / (2*depth_ref)
# Determine numbers of neighbouring blocks in dictionary
            if len(self.prisms[key].x_neigh) > 0:
                for kk in self.prisms[key].x_neigh:
                    k = np.where(keys == kk)[0]
# Add one to the diagonal positions of each block and set the position [i,k]
#     and the symmetric one [k,i] to -1
                    SS[i, i] += 1.
                    SS[k, k] += 1.
                    SS[k, i] = -1.
                    SS[i, k] = -1.
# Do the same for neighbouring blocks in Y direction
            if len(self.prisms[key].y_neigh) > 0:
                for kk in self.prisms[key].y_neigh:
                    k = np.where(keys == kk)[0]
                    SS[i, i] += 1.
                    SS[k, k] += 1.
                    SS[k, i] = -1.
                    SS[i, k] = -1.

# If inversion is done for at least two different parameter types, set the
#    smmothing parameters for the second parameter type
        z_fac[:] = 1.
        n0 = 0
        n1 = self.n_prisms
        if sus_inv:
            S[n0:n1, n0:n1] = SS/sigma_sus**2
            for i in range(len(keys)):
                S[i, :] *= z_fac[i]
                S[:, i] *= z_fac[i]
            n0 = n1
            n1 += self.n_prisms
        if rem_inv:
            S[n0:n1, n0:n1] = SS/sigma_rem**2
            for i in range(len(keys)):
                S[i+n0, :] *= z_fac[i]
                S[:, i+n0] *= z_fac[i]
            n0 = n1
            n1 += self.n_prisms
        if rho_inv:
            S[n0:n1, n0:n1] = SS/sigma_rem**2
            for i in range(len(keys)):
                S[i+n0, :] *= z_fac[i]
                S[:, i+n0] *= z_fac[i]
        return S

    def get_max_prisms(self, data, deriv, max_lim=0.1, width=10):
        """
        Find maxima in the matrix data defined as points that are larger than
        all other points in an area if width points around. For maxima with
        values larger than
        maxlim*(maximum_of_maxima - minimum_of_maxima)+minimum_of_maxima
        the prism is detected that has the strongest influence on the value of
        of the corresponding point (maximum derivative). A minimum of 3 prisms
        is searched for.
        If splitting the prism results in a prism smaller than the limit
        defined during initialization of the Class, do not integrate the
        corresponding prism in the outputted list

        Parameters
        ----------
        data : numpy 2D array of shape (ny,nx); float
            data set from which to find the extreme value.
        deriv : numpy 2D array (ndata x n_parameters), ndata = nx*ny
            Frechet matrix linking data to parameters (if several classes of
            data and/or parameters are used, pass only the part corresponding
            to one data/parameter class, i.e. pass a partial Frechet matrix)
        max_lim : float, optional Default is 0.1
            Prisms having the strongest influence on data values larger than
            the formula given above will be split.
        width : int, optional
            A maximum is recognized if the value at a point (i,j) is larger
            than or equal to all other values within an area of
            (i-width:i+width, j-width:j+width)


        Returns
        -------
        keys_max : list; int
            key values of all prisms lying beneath the maximum.

        """
        keys = list(self.prisms.keys())
        keys_max = []
        if len(data.shape) == 1:
            nx = len(data)
            ny = 1
        else:
            ny, nx = data.shape
        _, maxima = utils.get_extremes(abs(data), width)
        maxima = np.array(maxima)
        max_val = np.nanmin(data)
        min_val = np.nanmax(data)
        max_values = []
        max_keys = []
        for pos in maxima:
            npos = len(pos)
            if npos == 1:
                i = pos[0]
            else:
                i = pos[0]*nx + pos[1]
            p = np.argmax(abs(deriv[i, :]))
            key = keys[p]
            xpmn = self.prisms[key].x[0]
            xpmx = self.prisms[key].x[1]
            ypmn = self.prisms[key].y[0]
            ypmx = self.prisms[key].y[1]
            zpmn = self.prisms[key].z[0]
            zpmx = self.prisms[key].z[1]
            dx = (xpmx-xpmn)*0.5
            dy = (ypmx-ypmn)*0.5
            dz = (zpmx-zpmn)*0.5
# Check whether prims corresponding to this maximum can be split further
            if dx >= self.min_size_x or dy >= self.min_size_y or\
                    dz >= self.min_size_z:
                if npos == 1:
                    max_values.append(data[i])
                else:
                    max_values.append(data[pos[0], pos[1]])
                max_val = max(max_val, max_values[-1])
                min_val = min(min_val, max_values[-1])
                max_keys.append(key)
        max_values = np.array(max_values)
        max_keys = np.array(max_keys)
# Calculate the limiting value of maxima to be used for splitting of prisms
        max_test = min_val + (max_val - min_val)*max_lim
        index = np.argsort(max_values)
        max_keys = max_keys[index]
        max_values = max_values[index]
# If one local maximum is much larger than all the others, search for the
# value of the third largest local maximum and use this value as limit
#        npris = 0
        # for pos in maxima[index[::-1]]:
        #     if data[pos[0], pos[1]] < max_test and npris > 2:
        #         continue
        #     i = pos[0]*nx + pos[1]
        #     p = np.argmax(abs(deriv[i, :]))
        #     key = keys[p]
        #     xpmn = self.prisms[key].x[0]
        #     xpmx = self.prisms[key].x[1]
        #     ypmn = self.prisms[key].y[0]
        #     ypmx = self.prisms[key].y[1]
        #     zpmn = self.prisms[key].z[0]
        #     zpmx = self.prisms[key].z[1]
        #     dx = (xpmx-xpmn)*0.5
        #     dy = (ypmx-ypmn)*0.5
        #     dz = (zpmx-zpmn)*0.5
        #     if dx >= self.min_size_x or dy >= self.min_size_y or\
        #             dz >= self.min_size_z:
        #         keys_max.append(key)
        #         npris += 1
        for i, m in enumerate(max_values):
            if m >= max_test:
                keys_max.append(max_keys[i])
#            npris += 1
        return list(np.unique(np.array(keys_max)))

    def split(self, key):
        """
        Replace a prism by up to eight prisms, half the size in all directions.
        In each direction, the splitted size must not be smaller than the
        minimum size defined during initialization of the Class.

        Parameters
        ----------
        key : int
            key of the prism to be split in the dictionary.


        """
        sus_act = self.prisms[key].sus*4.*np.pi
        rem_act = self.prisms[key].rem
        rho_act = self.prisms[key].rho
        inc_act = self.prisms[key].inc
        dec_act = self.prisms[key].dec
        xpmn = self.prisms[key].x[0]
        xpmx = self.prisms[key].x[1]
        dx_act = xpmx-xpmn
        ypmn = self.prisms[key].y[0]
        ypmx = self.prisms[key].y[1]
        dy_act = ypmx-ypmn
        zpmn = self.prisms[key].z[0]
        zpmx = self.prisms[key].z[1]
        dz_act = zpmx-zpmn
        dx_p = dx_act/2.
        if dx_p < self.min_size_x:
            xp0 = [xpmn]
            xp1 = [xpmx]
        else:
            xp0 = [xpmn, xpmn+dx_p]
            xp1 = [xpmn+dx_p, xpmx]
        dy_p = dy_act/2.
        if dy_p < self.min_size_y:
            yp0 = [ypmn]
            yp1 = [ypmx]
        else:
            yp0 = [ypmn, ypmn+dy_p]
            yp1 = [ypmn+dy_p, ypmx]
        dz_act = zpmx-zpmn
        dz_p = dz_act/2.
        if dz_p < self.min_size_z:
            zp0 = [zpmn]
            zp1 = [zpmx]
        else:
            zp0 = [zpmn, zpmn+dz_p]
            zp1 = [zpmn+dz_p, zpmx]
        self.remove_prism(key)
        key_add = []
        for ix in range(len(xp0)):
            xpr = np.array([xp0[ix], xp1[ix]])
            for iy in range(len(yp0)):
                ypr = np.array([yp0[iy], yp1[iy]])
                for iz in range(len(zp0)):
                    zpr = np.array([zp0[iz], zp1[iz]])
                    self.add_prism(xpr, ypr, zpr, sus_act, rem_act, inc_act,
                                   dec_act, rho_act)
                    key_add.append(self.n_max)
        return key_add
