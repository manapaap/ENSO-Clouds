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

d_dy = shared.d_dy_mat(Nx, Ny, dy)
d2_dy2 = shared.d2_dy2_mat(Nx, Ny, dy)

Q_hat = np.fft.fft(Qtot, axis=1)

phi_operator = atms.phi_operator(Qtot, d_dy, d2_dy2)

# phi_operator, Q_hat = shared.apply_bc_chat(phi_operator, Q_hat)

phi_hat = shared.mat_operator_solve(phi_operator, Q_hat)
phi = (scipy.fft.ifft(phi_hat, axis=1)).real
u, v = atms.calc_vel(phi, dx, dy)



atms.plot_winds(u, v, 10, 5)

tau_x, tau_y = atms.calc_stress(u, v)

# u_ocean, v_ocean, h = ocean.calc_currents(dt, u_ocean, v_ocean, h,
#                                           tau_x, tau_y, dx, dy) 

# atms.plot_winds(u_ocean, v_ocean, 300, 150)


