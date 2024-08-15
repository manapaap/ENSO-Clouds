# -*- coding: utf-8 -*-
"""
File to analyze the climatalogical currents file crated by currents.py


Much smaller as the relevant dataset has been created!
"""


import xarray as xr
import numpy as np
from os import chdir
import cartopy.crs as ccrs
import matplotlib.pyplot as plt 
import metpy.calc as mpcalc
from metpy.units import units


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
cz_domain = [-30, 30, 120, -80 + 360]


def plot_waves(era5, every=20):
    """
    Plots wind barbs for climatological winds across the equatorial
    Pacific from ERA5 reanalysis.
    
    Plots every n'th observation as specified by "every".
    Assumes the data is already sliced to a single time step.
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    # Remove the time dimension if it's present
    u = np.asarray(era5.u)
    v = np.asarray(era5.v)
    
    # Create meshgrid for lon and lat
    lon2d, lat2d = np.meshgrid(era5.longitude, era5.latitude)
    
    # Subset the 2D arrays
    lon2d_subset = lon2d[::every, ::every].T
    lat2d_subset = lat2d[::every, ::every].T
    u_subset = u[::every, ::every]
    v_subset = v[::every, ::every]
       
    # Calculate wind magnitude
    mag = np.hypot(u_subset, v_subset)
    
    ax.set_title('Equatorial Pacific Climatological Currents')
    con = ax.quiver(lon2d_subset, lat2d_subset, u_subset, v_subset,
                    mag, cmap='viridis', transform=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.015, pad=0.04)


def calc_divergence(ocean):
    """
    Calculates the divergence of the surface currents using xarray's differentiation
    """
    u = ocean.u * units('m/s')
    v = ocean.v * units('m/s')
    
    # Calculate the divergence
    dudx = u.differentiate('longitude') / units('meter')
    dvdy = v.differentiate('latitude') / units('meter')
    
    divergence = dudx + dvdy
    
    return divergence


def plot_upwelling(ocean):
    """
    Plots the upwelling rate across ocean, as calculated from divergence
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    con = ax.contourf(ocean.longitude, ocean.latitude, ocean.w.T,
             origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')
        
    # Optional: Add coastlines, gridlines, etc.
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title('Equatorial Pacific Upwelling')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.02, pad=0.04)   
    
    plt.show()
    

def main():
    global ocean, upwell
    ocean = xr.open_dataset('misc_data/OSCAR_composite.nc')
    plot_waves(ocean, 10)
    
    H_1 = 50 # m- surface layer depth, a la Battisti
    ocean['w'] = H_1 * calc_divergence(ocean)  * units('m')
    
    # Remove extreme values
    cleaned = (np.asarray(ocean.w)).flatten()
    cleaned = cleaned[~np.isnan(cleaned)]
    max_val = np.quantile(cleaned, 0.995)
    min_val = np.quantile(cleaned, 0.005)
    # Replace the underlying data
    upwell_data = np.asarray(ocean.w.data)
    upwell_data[upwell_data > max_val] = max_val
    upwell_data[upwell_data < min_val] = min_val    
    
    ocean.w.data = upwell_data
    
    plot_upwelling(ocean)
    
    # Statistics
    print(f'Mean zonal velocity: {ocean.u.mean():.3f} m/s')
    print(f'Mean meriodional velocity: {ocean.v.mean():.3f} m/s')
    print(f'Mean upwelling velocity: {ocean.w.mean():.3f} m/s')
    
    
    
    
if __name__ == '__main__':
    main()


    
    
