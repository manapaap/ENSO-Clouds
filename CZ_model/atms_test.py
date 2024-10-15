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
#                               sigma=10)[::-1]
# temp -= p['Tbar']

# Get climatological convergence
conv = shared.interp_init_val('init_conds/conv.npy', Nx, Ny, smooth=True,
                              sigma=10)[::-1]
# we need to flip this to make it obey our sign conventions
# conv = None

# Idealized convergence?
conv = atms.idealized_conv(Nx, Ny)

# Initiate variables
temp = shared.idealized_sst_cz(Nx, Ny)
Q0 = atms.calc_Q0(temp)
Qtot = -Q0

# Let's try the iteration
# Invariants
phi_operator = atms.phi_operator(Nx, Ny, dx, dy)
v_operator = atms.v_operator(Nx, Ny, dx, dy)
beta_y = atms.get_beta_y(Nx, Ny)    
# Loop conditions
u_prev = np.nan
v_prev = np.nan
# Tolerance for convergence
tol = 0.001

for n in range(50):
    # This will skip applying the BCs to the operators after the first iter
    # Not perfect but removes 80% of the redundancy
    LHS_v = atms.v_rhs(Qtot, dx, dy)
    v_operator, LHS_v = shared.apply_bc(v_operator, LHS_v, skip_op=bool(n))
    v = shared.mat_operator_solve(v_operator, LHS_v)
    
    # Now solve for phi 
    LHS_phi = atms.phi_rhs(Qtot, v, dx, dy)
    phi_operator, LHS_phi = shared.apply_bc(phi_operator, LHS_phi)
    phi = shared.mat_operator_solve(phi_operator, LHS_phi)
    # Simple for u
    u = -(p['epsilon'] * v + np.gradient(phi, dy, axis=0)) / beta_y
    # Update Qtot and continue the loop
    Q1 = atms.calc_Q1(u, v, dx, dy, conv=conv.copy())
    Qtot = -Q0 - Q1
    # Loop break conditions
    if abs(np.mean(u_prev - u) + np.mean(v_prev - v)) < tol:
        break
    else:
        u_prev = u
        v_prev = v
    
atms.plot_winds(u, v, 5, 2)

tau_x, tau_y = atms.calc_stress(u, v)

# u_ocean, v_ocean, h = ocean.calc_currents(dt, u_ocean, v_ocean, h,
#                                           tau_x, tau_y, dx, dy) 

# atms.plot_winds(u_ocean, v_ocean, 300, 150)


