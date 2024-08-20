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
from os import chdir


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')


from CZ_model.standard_vals import get_params


# Global parameter values
p = get_params()


def calc_Q0(T_grid):
    """
    Calculates Q0 "first guess" of anomalous heating due to anomalous evaporat
    
    takes SST grid, returns Q0 grid
    """
    first_term = p['gamma'] * p['ca']**2 * np.sqrt(2 * p['beta'] / p['ca'])
    second = T_grid * (p['Tref'] / p['Tbar'])**2
    inside_exp = p['b'] * (p['Tref']**-1 - p['Tbar']**-1)
    
    return first_term * second * np.exp(inside_exp)


def positive(grid):
    """
    Returns value if positive, zero if negative
    """
    grid[grid < 0] = 0
    return grid


def calc_Q1(u, v, dx, dy):
    """
    Calculates the "anomalous" term for moisture convergence
    """
    # Compute partial derivatives using central differences
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Calculate the divergence
    divergence = du_dx + dv_dy
    
    return positive(-p['alpha_eff'] * divergence)


def calc_Q0_CZ(T_grid):
    """
    Calculates Q0 using the "original" CZ method rather than Battisti's modidied
    one
    """
    exp_term = (p['Tbar'] - (273.15 + 30)) / (273.15 + 16.7)
    
    return 0.0031 * T_grid * np.exp(exp_term)


def calc_div(u, v, dx, dy):
    """
    Calculates the divergence of the flow
    """
    # Compute partial derivatives using central differences
    du_dx = np.gradient(u, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    
    # Calculate the divergence
    divergence = du_dx + dv_dy
    
    return divergence


def calc_phi(u, v, Q0, Q1, dx, dy):
    """
    Calculates phi from A3 in battisti. Uses:
        
        Q0 from current SST
        u, v from previous iteration (prev time step or prev iteration)
        Q1 from prev iteration (zero if first iteration)
    """
    div = calc_div(u, v, dx, dy)
    
    intermed = -Q0 - (p['ca']**2) * div + Q1
    
    return intermed / p['epsilon']
    

def plot_winds(u, v, every_x=40, every_y=20):
    """
    PLots wind barbs for our domain
    """
    Nx = u.shape[1]
    Ny = v.shape[0]
    
    y_dim = np.linspace(-p['Ly'] / 2, p['Ly'] / 2, Ny)[::every_y]
    x_dim = np.linspace(0, p['Lx'], Nx)[::every_x]
    
    u_cut = u[::every_y, ::every_x]
    v_cut = v[::every_y, ::every_x]
    mag = np.hypot(u_cut, v_cut)
    
    x_dim, y_dim = np.meshgrid(x_dim, y_dim)
    
    plt.quiver(x_dim, y_dim, u_cut, v_cut, mag)
    plt.title('Anomalous Winds')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    plt.show()
    

def calc_vel(phi, dx, dy):
    """
    Re-calculates the values of u, v depending on the pressure (perturbation?)
    recieved from phi
    
    Remember, numpy indexing means that y is positive downwards
    """
    global beta_y, dphi_dx, dphi_dy
    Ny = phi.shape[0]
    Nx = phi.shape[1]
    
    dphi_dx = -np.gradient(phi, dx, axis=1)
    # This shoul be negative but I think numpy direction conventions
    # are making this weird
    dphi_dy = -np.gradient(phi, dy, axis=0)
    
    # Reverse order so beta_y increases northwards
    beta_y = np.linspace(-1000 * p['Ly'] / 2, 1000 * p['Ly'] / 2, Ny) *\
        p['beta']
    # Stack this in the x-dimention to get the shape we want
    _, beta_y = np.meshgrid(np.ones(Nx), beta_y)
    
    # Calculate u
    const = p['epsilon'] * dphi_dx + beta_y * dphi_dy
    u = const / (p['epsilon']**2 + beta_y**2)
    # Calculate v
    v = (dphi_dy - beta_y * u) / p['epsilon']
    
    return u, v
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




