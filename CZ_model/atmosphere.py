# -*- coding: utf-8 -*-
"""
Atmospheric component of CZ model

This file will provide either functions or an object representing the
atmospheric component of a CZ model (TBD). The input will be the SST mesh,
and the output will be the steady state winds and the resultant stress

Solution method: TBD (probably spectral)
"""


import numpy as np
import matplotlib.pyplot as plt


# parameter values- as per Cane-Zebiak or Battisti
beta = 1.6 * 10**4 # m-1 s-1 I believe
epsilon = 1.576 * 10**-5 # s-1, a la Battisti
gamma = 1.6 # m K s-1
alpha = 0.75
rho = 1.275
Tref = 303 # K 
Tbar = 273 + 26.5 # K, need to update using data
b = 5400 # K


class Atmosphere:
    def __init__(self, dt, Nx, Ny, ua, t, Ti):
        """
        Initializes the atmospheric model with given time step, 
        resolution in x/y,
        function describing the initial wind anomaly, a timeline for the
        duration of this anomaly (in days), \
        and an initial temperature distribution
        """
        if Ti.shape != (Ny, Nx):
            print('ERROR: Grids do not match')
        
        self.u = np.zeros((Ny, Nx))
        self.v = np.zeros((Ny, Nx))


