# -*- coding: utf-8 -*-
"""
SST, SLP, and mean climatological winds for the CZ model

From ERA5 reanalysis
"""


import xarray as xr
import numpy as np
from os import chdir
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

cz_domain = [-30, 30, 120, -80 + 360]


def plot_winds(era5, every=20):
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
    u = np.asarray(era5.u10)
    v = np.asarray(era5.v10)
    
    # Create meshgrid for lon and lat
    lon2d, lat2d = np.meshgrid(era5.longitude, era5.latitude)
    
    # Subset the 2D arrays
    lon2d_subset = lon2d[::every, ::every]
    lat2d_subset = lat2d[::every, ::every]
    u_subset = u[::every, ::every]
    v_subset = v[::every, ::every]
       
    # Calculate wind magnitude
    mag = np.hypot(u_subset, v_subset)
    
    ax.set_title('Equatorial Pacific 10 m winds')
    con = ax.quiver(lon2d_subset, lat2d_subset, u_subset, v_subset,
                    mag, cmap='viridis', transform=ccrs.PlateCarree())
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.015, pad=0.04)
    

def plot_scalar(era5, var, title):
    """
    Contour plot of sea level pressure across the equatorial pacific
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    if var == 'msl':
        # Get data in HPa
        era5[var] /= 100
    
    con = ax.contourf(era5.longitude, era5.latitude, era5[var],
             origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')
    #ax.contour(era5.longitude, era5.latitude, era5[var],
    #         origin='lower', transform=ccrs.PlateCarree(), colors='black')
        
    # Optional: Add coastlines, gridlines, etc.
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title('Equatorial Pacific Climatological ' + title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.02, pad=0.04)   
    
    plt.show()


def calc_convergence(ds):
    # Get units in meters
    dx = 111320 * ds['longitude'].diff('longitude')
    dy = 111320 * ds['latitude'].diff('latitude')
    
    # Calculate partial derivatives in-place, without creating large intermediate arrays
    du_dx = ds['u10'].diff('longitude') / dx
    dv_dy = ds['v10'].diff('latitude') / dy
    
    # Adding derivatives directly, minimizing memory overhead
    divergence = du_dx + dv_dy
    return -divergence


def main():
    global era5
    # Should be able to do all the processing in main since there isn't too
    # much here
    era5 = xr.open_dataset('era5_reanal/era5_reanal_new.nc')
    # merge 5 and 5T
    # era5 = era5.sel(expver=1).combine_first(era5.sel(expver=5))
    # Subset to paciic
    min_lat, max_lat, min_lon, max_lon = cz_domain
    era5 = era5.sel(latitude=slice(max_lat, min_lat), 
                    longitude=slice(min_lon, max_lon))
    # Mean across time - climatology
    era5 = era5.mean(dim='time', skipna=True)
    # Low-level convergence
    era5['conv'] = calc_convergence(era5)
    
    # Mean across domain
    print(f'Mean Zonal Windspeed: {float(era5.u10.mean()):.2f} m/s')
    print(f'Mean Meridional Windspeed: {float(era5.v10.mean()):.2f} m/s')
    print(f'Mean Domain SST: {float(era5.sst.mean()):.2f} K')
    
    # Plots
    plot_winds(era5, every=20)
    plot_scalar(era5, 'msl', 'Sea Level Pressure')
    plot_scalar(era5, 'sst', 'Sea Surface Temperature')
    plot_scalar(era5, 'conv', 'Low Level Convergence')
    
    # Let's write these fields as a numpy array so they can be used as initial
    # conditions; we want SLP, SST
    np.save('init_conds/sst.npy', era5.sst.data)
    np.save('init_conds/slp.npy', era5.msl.data)
    np.save('init_conds/u.npy', era5.u10.data)
    np.save('init_conds/v.npy', era5.v10.data)  
    np.save('init_conds/conv.npy', era5.conv.data)  

if __name__ == '__main__':
    main()
