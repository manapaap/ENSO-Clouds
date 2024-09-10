# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:20:54 2024

@author: aakas
"""

import numpy as np
from os import chdir

chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

import CZ_model.atmosphere as atms
import CZ_model.standard_funcs as shared
import CZ_model.ocean as ocean
import scipy.fft


p = shared.get_params()


Nx = 150
Ny = 50


dx = p['Lx'] / Nx
dy = p['Ly'] / Ny
dt = 24 * 3600 # 1 day

# Initiate ocean with zeros
u_ocean = np.zeros((Ny, Nx))
v_ocean = np.zeros((Ny, Nx))

h = np.zeros((Ny, Nx))

# temp = shared.interp_init_val('init_conds/sst.npy', Nx, Ny, smooth=True,
#                               sigma=5)
# temp -= p['Tbar']
temp = shared.idealized_sst_cz(Nx, Ny)
u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))

Q0 = atms.calc_Q0(temp)
Q1 = atms.calc_Q1(u, v, dx, dy)
Qtot = -Q0 + Q1

# Operator matrices
d_dx = shared.d_dx_mat(Nx, Ny, dx)
d_dy = shared.d_dy_mat(Nx, Ny, dy)
d2_dy2 = shared.d2_dy2_mat(Nx, Ny, dy)
d2_dx2 = shared.d2_dx2_mat(Nx, Ny, dx)
eye = np.diag(np.ones(Ny * Nx))

beta_y = atms.get_beta_y(Nx, Ny)

LHS_v = -p['epsilon'] * np.gradient(Qtot, dy, axis=0) / p['ca']**2
# include the PV-heating term?
LHS_v += beta_y * np.gradient(Qtot, dx, axis=1) / p['ca']**2

# Construct the v-operator piecewise
linear = (beta_y**2 + p['epsilon']**2) / p['ca']**2
linear = eye * linear.reshape(-1) * p['epsilon']
lap = -p['epsilon'] * (d2_dy2 + d2_dx2)
first = -p['beta'] * d_dx

v_operator = linear + lap + first

v_operator, LHS_v = shared.apply_bc(v_operator, LHS_v)

v = shared.mat_operator_solve(v_operator, LHS_v)

# Similar approach for phi
phi_operator = beta_y.reshape(-1) * d_dx - p['epsilon'] * d_dy
LHS_phi = (beta_y**2 + p['epsilon']**2) * v

phi_operator, LHS_phi = shared.apply_bc(phi_operator, LHS_phi)
phi = shared.mat_operator_solve(phi_operator, LHS_phi)

# Not very successful, try to solve for u
u_operator = beta_y.reshape(-1) * d_dx - p['epsilon'] * d_dy
LHS_u = -p['beta'] * v - beta_y * np.gradient(v, dy, axis=0) -\
    p['epsilon'] * np.gradient(v, dx, axis=1)

u_operator, LHS_u = shared.apply_bc(u_operator, LHS_u)
u = shared.mat_operator_solve(u_operator, LHS_u)


# Alternate apppraoch for phi
# LHS_phi = (1 - beta_y**2 / p['epsilon']**2) * Qtot - 2 * p['ca']**2 *\
#    p['beta'] * beta_y * v * p['epsilon']**2
# phi_operator = (p['epsilon']**2 - beta_y**2).reshape(-1) * eye / p['epsilon'] -\
#     (p['ca']**2 * p['beta'] / p['epsilon']**2) * d_dx -\
#     (p['ca']**2 / p['epsilon']) * (d2_dy2 + d2_dx2)
    
# phi_operator, LHS_phi = shared.apply_bc(phi_operator, LHS_phi)
# phi = shared.mat_operator_solve(phi_operator, LHS_phi)



atms.plot_winds(u, v, 5, 2)

tau_x, tau_y = atms.calc_stress(u, v)

# u_ocean, v_ocean, h = ocean.calc_currents(dt, u_ocean, v_ocean, h,
#                                           tau_x, tau_y, dx, dy) 

# atms.plot_winds(u_ocean, v_ocean, 300, 150)


