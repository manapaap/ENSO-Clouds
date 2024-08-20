# -*- coding: utf-8 -*-
"""
Initial conditions and Base values script for CZ model

Returns the initial conditions we ask for (e.g. SST) along with the
climatological data for the thermocline. Basically the stuff that varies in 
space

Also instantiates the parameter dictionary since I don't want to muck about
with that in every file. '

Interpolates to a regular grid as defined by Nx/Ny
"""


import xarray as xr
import numpy as np


def idealized_sst(Nx, Ny):
    """
    idealized initial condition for SST: a gaussian in y and a sine wave in 
    x
    """
    L_x = 17811 * 1000 # m
    L_y = 6672 * 1000 # m (online lat/lon calculator)
    
    # Calibrate Nx/Ny to be around a 3km grid box
    y_dim = np.linspace(0, L_y, Ny)
    x_dim = np.linspace(0, L_x, Nx)
    
    cent = L_y / 2
    sig = L_y / 4
    gaus_y = gaussian(y_dim, cent, sig)
    
    sin_x = (1.5) * np.sin(2 * np.pi * x_dim / L_x)
    
    temp = np.outer(gaus_y, sin_x)
    
    return temp + 299.21


def idealized_winds(Nx, Ny):
    """
    idealized initial condition for winds: a gaussian in u and nothing in v
    """
    L_x = 17811 * 1000  # m
    L_y = 6672 * 1000   # m
    
    # Calibrate Nx/Ny to be around a 3km grid box
    x_dim = np.linspace(0, L_x, Nx)
    y_dim = np.linspace(0, L_y, Ny)
    x_base = np.ones(Nx)
    
    cent = L_y / 2
    sig = L_y / 4
    gaus_y = gaussian(y_dim, cent, sig)
    
    u = np.outer(gaus_y, x_base)
    # Normalize the domain so that its maximum value is 1.5
    max_value = np.max(u)
    u = (u / max_value) * -1.5
    
    v = np.zeros((Ny, Nx))
    
    return u, v


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)


def get_params():
    """
    Returns the parameter values used by the CZ model calculation
    
    Inefficnent way to store things but I don't want to write this out over
    and over
    """
    params = dict()
    
    # Atmosphere parameter values- as per Cane-Zebiak or Battisti
    params['beta'] = 2.28 * 10**-11 # m-1 s-1 I believe
    params['epsilon'] = 1.576 * 10**-5 # s-1, a la Battisti
    params['gamma'] = 1.6 # m K s-1
    params['alpha_eff'] = 0.75
    params['rho_air'] = 1.275 # kg m-3
    params['Tref'] = 303 # K 
    params['Tbar'] = 299.21 # K, need to update using data
    params['b'] = 5400 # K
    params['ca'] = 60 # m s-1

    # Ocean parameters
    params['H0'] = 150 # m
    params['g'] = 5.6 * 10**-2 # m s-2 reduced gravity
    params['c0'] = 2.89 # m s-1
    params['r'] = 2.5 # years
    params['Cd'] = 3.2 * 10**-3
    params['rho_ocean'] = 1.026 * 10**3 # kg m-3
    params['H1'] = 50 # m
    params['delta'] = 0.75
    params['alpha_damp'] = 125 # days
    params['r_s'] = 2 # days

    # Domain size
    params['Lx'] = 17811 * 1000 # m
    params['Ly'] = 6672 * 1000 # m (online lat/lon calculator)
    
    # Meterological params
    params['u_a'] = -3.45 # m/s
    params['Tbar'] = 299.21 # K
    params['u_o'] = -0.045 # m/s
    params['w'] = 0.025 # m/s
    
    # domain
    params['domain'] = [-30, 30, 120, -80 + 360]
    
    return params