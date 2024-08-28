# -*- coding: utf-8 -*-
"""
Thermocline Depth data from 

https://www.ncei.noaa.gov/access/metadata/landing-page
/bin/iso?id=gov.noaa.nodc:0205198

for the sake of my CZ model

Need the mean specified climatological thermocline
"""


import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from os import chdir
from scipy.ndimage import gaussian_filter1d


chdir('C:/Users/aakas/Documents/ENSO-Clouds/')


cz_domain = [-2, 2, 120, -80 + 360]


def load_ocean_data(fpath='ocean_props/data/WOA18annAll025.nc', 
                    domain=cz_domain):
    """
    Loads the ocean data and masks it to the CZ model domain
    """
    min_lat, max_lat, min_lon, max_lon = domain
    
    ocean_all = xr.load_dataset(fpath)
    
    # Rename Latitude and Longitude to lat and lon
    ocean_all = ocean_all.rename({'Latitude': 'lat', 'Longitude': 'lon'})
    
    # Convert lon from -180 to 180 to 0 to 360 and create a new DataArray for it
    transformed_lon = np.where(ocean_all['lon'] < 0, ocean_all['lon'] + 360,
                               ocean_all['lon'])
    transformed_lon = xr.DataArray(transformed_lon, dims=ocean_all['lon'].dims,
                                   coords=ocean_all['lon'].coords)
    
    # Update the dataset with the transformed longitude
    ocean_all = ocean_all.assign_coords(lat=ocean_all['lat'],
                                        lon=transformed_lon)
    
    # Sort the dataset along the 'lon' dimension to ensure monotinically
    # increasing values
    ocean_all = ocean_all.sortby('lon')
    
    # Correct coordinate orientation and select the region
    cz_ocean = ocean_all.sel(lat=slice(min_lat, max_lat), 
                             lon=slice(min_lon, max_lon))

    return cz_ocean


def plot_thermocline_map(data, domain=cz_domain, title=''):
    """
    Plots the climatological thermocline depth from given data.
    """
    min_lat, max_lat, min_lon, max_lon = domain  
    
    # Set up the plot with a central longitude of 180 for the PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    
    # Plot the data with the correct transform, central_longitude needs to be consistent
    con = ax.contourf(data.lon, data.lat, data.T,
                extent=[min_lon, max_lon, min_lat, max_lat], 
             origin='lower', transform=ccrs.PlateCarree(), cmap='viridis')
    
    # Set the extent based on domain or data coordinates
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    # Optional: Add coastlines, gridlines, etc.
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title(title)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    fig = plt.gcf()
    fig.colorbar(con, fraction=0.02, pad=0.04)   
    
    plt.show()


def gaussian_smoothing(data, sigma):
    """
    Smooth data using Gaussian smoothing.
    """
    return gaussian_filter1d(data, sigma=sigma)


def plot_thermocline(mean_cline, title='', ylabel=''):
    """
    Plots the mean thermocline depth along the longitude axis
    
    Also plots (and returns) a smoothed value using a gaussian filter
    """
    smooth = gaussian_smoothing(mean_cline.data, 15)
    
    plt.figure(np.random.randint(0, 100000))
    plt.plot(mean_cline.lon, mean_cline.data, label='Original')
    plt.plot(mean_cline.lon, smooth, label='Gaussian Smooth', color='red', 
             linestyle='dashed')
    plt.grid()
    plt.xlabel('Longitude')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    
    return smooth


def main():
    # Remove the land fill value   
    global thermocline
    ocean = load_ocean_data()
    thermocline = ocean['ThermoclineDepth'].where(ocean['ThermoclineDepth'] >= 0,
                                                  np.nan)
    thermo_grad = ocean['MeanThermoclineGradient'].\
            where(ocean['MeanThermoclineGradient'] >= -1000, np.nan)
    
    plot_thermocline_map(thermocline,
                         title='Pacific Climatological Thermocline Depth')
    plot_thermocline_map(thermo_grad, 
                         title='Pacific Climatological Thermocline Gradient')
    
    # Get mean depth along longitude
    mean_cline = thermocline.mean(dim='lat', skipna=True)
    mean_grad = thermo_grad.mean(dim='lat', skipna=True)
    
    # Using the smoothed value makes sense as we remove the impact of landmasses
    # and the like
    plot_thermocline(mean_cline, title='Mean Thermocline Depth', 
                     ylabel='Depth (m)')
    plot_thermocline(mean_grad, title='Mean Thermocline Gradient',
                     ylabel='dT/dz (K/m)')
    
    # TODO: Run this buy mike and produce a function that will yield this array
    # when necessary in the larger CZ model
    
    
if __name__ == '__main__':
    main()
