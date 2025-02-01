# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:27:45 2024

@author: Hermann
"""

import numpy as np

class Earth_mag():
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
        self.cdci = self.cde*self.cie
        self.sdci = self.sde*self.cie
        self.eh = self.f*self.cie
        self.ex = self.f*self.cdci
        self.ey = self.f*self.sdci
        self.ez = self.f*self.sie
        return True

