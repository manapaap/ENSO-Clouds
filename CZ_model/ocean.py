# -*- coding: utf-8 -*-
"""
OCeanic component of CZ model

functions to solve for the oceanic variables one by one
"""

import numpy as np
import matplotlib.pyplot as plt
from os import chdir


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')


from CZ_model.standard_funcs import get_params


# Global parameter values
p = get_params()


def get_beta_y(Nx, Ny):
    """
    Returns beta_y meshgrid for given Nx, Ny
    """
    beta_y = np.linspace(-p['Ly'] / 2, p['Ly'] / 2, Ny) *\
        p['beta']
    # Stack this in the x-dimention to get the shape we want
    _, beta_y = np.meshgrid(np.ones(Nx), beta_y)

    return beta_y


def calc_currents(dt, u, v, h, tau_x, tau_y, dx, dy):
    """
    Calculates the net current in both layers of the ocean from previous iteration
    
    returns u_new, v_new, h_new
    """
    Ny = u.shape[0]
    Nx = u.shape[1]
    
    beta_y = get_beta_y(Nx, Ny)
    dh_dx = np.gradient(h, dx, axis=1)
    dh_dy = np.gradient(h, dy, axis=0)
    
    h_new = h - dt * (p['r'] * h + p['H0'] * calc_div(u, v, dx, dy))
    
    u_new = u + dt * (-p['g'] * dh_dx + (tau_x / (p['H0'] * p['rho_ocean']))\
                      -p['r'] * u + beta_y * v)
    v_new = (-p['g'] * dh_dy + (tau_y / (p['H0'] * p['rho_ocean'])) -\
             beta_y * u_new) / p['r']
    
    return u_new, v_new, h_new


def calc_div(u, v, dx, dy):
    """
    Calculates the divergence of the flow (copied from atmos module)
    """
    # Compute partial derivatives using central differences
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Calculate the divergence
    divergence = du_dx + dv_dy
    
    return divergence