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
    

def plot_slp(era5):
    """
    Contour plot of sea level pressure across the equatorial pacific
    """
    plt.figure()
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    
    con = ax.contourf(era5.longitude, era5.latitude, era5.msl / 100,
             origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')
        
    # Optional: Add coastlines, gridlines, etc.
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title('Equatorial Pacific Climatological Sea Level Pressure')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.02, pad=0.04)   
    
    plt.show()

    
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
    
    # Mean across domain
    print(f'Mean Zonal Windspeed: {float(era5.u10.mean()):.2f} m/s')
    print(f'Mean Meridional Windspeed: {float(era5.v10.mean()):.2f} m/s')
    print(f'Mean Domain SST: {float(era5.sst.mean()):.2f} K')
    
    # Plots
    plot_winds(era5, every=20)
    plot_slp(era5)
    

if __name__ == '__main__':
    main()
