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
import time


p = shared.get_params()


Nx = 300
Ny = 100


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

phi, u, v = atms.solve_atmos_fourier(Qtot, dy)


# d2_dx2 = shared.d2_dx2_mat(Nx, Ny, dx)
# d2_dy2 = shared.d2_dy2_mat(Nx, Ny, dy)
# epsilon_mat = np.diag(p['epsilon'] * np.ones(Nx * Ny), k=0)

# phi_operator = epsilon_mat - (p['ca']**2 * p['epsilon']**-1) * (d2_dx2 + d2_dy2)

# phi_operator, Qtot = shared.apply_bc(phi_operator, -Q0 + Q1)



# phi = shared.mat_operator_solve(phi_operator, Qtot, GPU=False)
# u, v = atms.calc_winds_iter(u, v, phi, dx, dy)

# vort_like = atms.vorticity_like(u, v, dx, dy)

# Qtot += vort_like
# Qtot = shared.apply_bc_vect(Qtot)


atms.plot_winds(u, v, 15, 7)

tau_x, tau_y = atms.calc_stress(u, v)

u_ocean, v_ocean, h = ocean.calc_currents(dt, u_ocean, v_ocean, h,
                                          tau_x, tau_y, dx, dy) 

# atms.plot_winds(u_ocean, v_ocean, 300, 150)


