# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:20:54 2024

@author: aakas
"""

import numpy as np
from os import chdir

chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

import CZ_model.atmosphere as atms
import CZ_model.standard_vals as shared
import CZ_model.ocean as ocean


p = shared.get_params()


Nx = 6000
Ny = 2000

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
u0, v0 = shared.idealized_winds(Nx, Ny)

Q0 = atms.calc_Q0(temp)
Q0_cz = atms.calc_Q0_CZ(temp)
Q0_ff = atms.calc_Q0_ff(temp)
Q1 = atms.calc_Q1(u0, v0, dx, dy)

phi = atms.calc_phi(u0, v0, Q0, Q1, dx, dy)

u, v = atms.calc_vel(phi, dx, dy)
atms.plot_winds(u, v, 300, 150)

tau_x, tau_y = atms.calc_stress(u, v)

u_ocean, v_ocean, h = ocean.calc_currents(dt, u_ocean, v_ocean, h,
                                          tau_x, tau_y, dx, dy) 

atms.plot_winds(u_ocean, v_ocean, 300, 150)