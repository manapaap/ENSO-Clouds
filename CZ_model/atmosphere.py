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
import scipy.fft


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
    div = calc_div(u, v, dx, dy)
    
    return positive(-p['alpha_eff'] * div * p['ca']**2)


def calc_Q0_CZ(T_grid):
    """
    Calculates Q0 using the "original" CZ method rather than Battisti's modidied
    one
    """
    exp_term = (p['Tbar'] - (273.15 + 30)) / (273.15 + 16.7)
    
    return 0.0031 * T_grid * np.exp(exp_term)


def calc_Q0_ff(T_grid):
    """
    Calculates Q0 using the moditified, non iterated method from Geng and Jin
    2023 and not the original one
    """
    a_q = 7.6 * 10**-6 # m2s-3
    b_q = 0.5
    
    exp_term = 0.0031 * np.exp(b_q * (p['Tbar'] - (273.15 + 30)) / (273.15 + 16.7)) 
    
    lin_anomaly = (b_q * T_grid) + (0.5 * (b_q * T_grid)**2) +\
        ((6**-1) * (b_q * T_grid)**3)
    
    return exp_term * lin_anomaly


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
    cbar = plt.colorbar()
    cbar.ax.set_title('Windspeed (m/s)')
    
    fig = plt.gcf()
    fig.set_size_inches(7.5, 3)
    
    plt.show()
    
    
def plot_scalar(field, title='', scale=''):
    """
    plots scalar field over domain
    """
    Nx = field.shape[1]
    Ny = field.shape[0]
    
    y_dim = np.linspace(-p['Ly'] / 2, p['Ly'] / 2, Ny)
    x_dim = np.linspace(0, p['Lx'], Nx)
        
    x_dim, y_dim = np.meshgrid(x_dim, y_dim)
    
    plt.contourf(x_dim, y_dim, field)
    cbar = plt.colorbar()
    cbar.ax.set_title(scale)
    plt.contour(x_dim, y_dim, field, colors='black')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid()
    
    fig = plt.gcf()
    fig.set_size_inches(7.5, 3)
    
    plt.show()    


def get_beta_y(Nx, Ny):
    """
    Returns beta_y meshgrid for given Nx, Ny
    """
    beta_y = np.linspace(-p['Ly'] / 2, p['Ly'] / 2, Ny) *\
        p['beta']
    # Stack this in the x-dimention to get the shape we want
    _, beta_y = np.meshgrid(np.ones(Nx), beta_y)

    return beta_y

    
def calc_vel(phi, dx, dy):
    """
    Re-calculates the values of u, v depending on the pressure (perturbation?)
    recieved froirm phi
    
    Remember, numpy indexing means that y is positive downwards
    """
    Ny = phi.shape[0]
    Nx = phi.shape[1]
    
    dphi_dx = -np.gradient(phi, dx, axis=1)
    # This shoul be negative but I think numpy direction conventions
    # are making this weird
    dphi_dy = -np.gradient(phi, dy, axis=0)
    
    beta_y = get_beta_y(Nx, Ny)
    
    # Calculate u
    const = p['epsilon'] * dphi_dx + beta_y * dphi_dy
    u = const / (p['epsilon']**2 + beta_y**2)
    # Calculate v
    v = (dphi_dy - beta_y * u) / p['epsilon']
    
    return u, v


def calc_winds_iter(u, v, phi, dx, dy):
    """
    uses Peter's suggestion of Jacobi-style iteration to calculate u, v
    """
    beta_y = get_beta_y(*u.shape[::-1])    
    
    dphi_dx = np.gradient(phi, dx, axis=1)
    dphi_dy = np.gradient(phi, dy, axis=0)
    
    # Update u/v/phi
    u = (beta_y * v - dphi_dx) / p['epsilon']
    v = (-beta_y * u - dphi_dy) / p['epsilon']

    return u, v   
    

def calc_stress(u, v):
    """
    Uses quadratic drag to calculate and return wind stress in x and y
    """
    speed = np.sqrt(u**2 + v**2)
    
    tau_x = p['rho_air'] * p['Cd'] * u * speed
    tau_y = p['rho_air'] * p['Cd'] * v * speed
    
    return tau_x, tau_y
    
    
def vorticity_like(u, v, dx, dy):
    """
    Calculates the vorticity-like terms that modify Qt during each iteration
    """
    beta_y = get_beta_y(*u.shape[::-1])
    
    term_one = -p['ca']**2 * beta_y * np.gradient(v, dx, axis=1)
    term_two = -p['ca']**2 * np.gradient(beta_y * u, dy, axis=0)
    
    return (term_one + term_two) / p['epsilon']
    
    
def solve_atmos_fourier(Qt, dy):
    """
    Solves for u, v, phi using a fourier transform in x and finite differencing
    in Y. The boundary condition to be considered is u, v, phi = 0 
    outside the box
    """
    beta_y = get_beta_y(*Qt.shape[::-1])
    
    Qt_fft = scipy.fft.fft(Qt, axis=1)
    
    Ny, Nx = Qt.shape
    
    # get the wavenumbers and turn into a mesh
    kx = scipy.fft.fftfreq(Nx, d=p['Lx'] / Nx) 
    kx = np.outer(np.ones(Ny), kx)
    
    phi_hat = np.zeros_like(Qt)
    u_hat = np.zeros_like(Qt)
    v_hat = np.zeros_like(Qt)
    
    for j in range(Ny):
        if j == 0:
            phi_hat[j, :] = 0
            u_hat[j, :] = 0
            v_hat[j, :] = 0
        elif j == 1:
            phi_hat[j] = - 2 * dy * (beta_y[j - 1] * u_hat[j - 1] +\
                                                 p['epsilon'] * v_hat[j]) 
            v_hat[j] = - 2 * dy * (p['epsilon'] * phi_hat[j] +\
                                                p['ca']**2 * 1j * kx[j] *\
                                                    u_hat[j] - Qt_fft[j])
            u_hat[j, :] = 0
        else:
            phi_hat[j] = phi_hat[j - 2] -\
                            2 * dy * (beta_y[j - 1] * u_hat[j - 1] +\
                            p['epsilon'] * v_hat[j]) 
            v_hat[j] = v_hat[j - 2] - 2 * dy * (p['epsilon'] * phi_hat[j] +\
                                                p['ca']**2 * 1j * kx[j] *\
                                                    u_hat[j] - Qt_fft[j])
            u_hat[j] = (beta_y[j] * v_hat[j] - 1j * kx[j] * phi_hat[j]) /\
                p['epsilon']
    
    # Inverse the fft
    phi = (scipy.fft.ifft(phi_hat, axis=1)).real
    u = (scipy.fft.ifft(u_hat, axis=1)).real
    v = (scipy.fft.ifft(v_hat, axis=1)).real
    
    return phi, u, v
    
    
    
def phi_operator(Qt, d_dy, d2_dy2):
    """
    Constructs the linear operator that solves for phi each iteration
    
    Qt is just taken to get the shape of the system. 
    """
    Ny, Nx = Qt.shape
    beta_y = get_beta_y(Nx, Ny)
    
    # get the wavenumbers and turn into a mesh
    kx = scipy.fft.fftfreq(Nx, d=p['Lx'] / Nx) 
    kx = np.outer(np.ones(Ny), kx) 
    
    # This denominator is seen often so we will store it
    beta_ep_sq = (beta_y**2 + p['epsilon']**2)**-1
    
    # Lets do this piece-by-piece to help comprehension
    zero_der = (p['ca']**2 + kx**2 / p['epsilon']) - (beta_y**2 * beta_ep_sq *\
            (p['ca']**2 + kx**2 / p['epsilon'])) + p['epsilon']
    zero_der += -2 * p['ca']**2 * p['beta'] * beta_y**2 * 1j * kx * beta_ep_sq**2
    zero_der += p['ca']**2 * p['beta'] * 1j * kx * beta_ep_sq
    # Now for the first derivative
    first_der = -beta_y * 1j * kx * p['ca']**2 * beta_ep_sq
    first_der += 2 * p['ca']**2 * p['epsilon'] * p['beta'] * beta_y * beta_ep_sq**2
    first_der += p['ca']**2 * beta_y * 1j * beta_ep_sq
    # FInal component!
    sec_der = p['ca']**2
    
    # Time to add these together
    operator = np.diag(np.ones_like(d_dy)) * zero_der + d_dy * first_der +\
        d2_dy2 * sec_der
    
    return operator
    
    
    
    
    




