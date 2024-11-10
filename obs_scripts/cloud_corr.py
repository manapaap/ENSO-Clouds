# -*- coding: utf-8 -*-
"""
Cloud Correlations and ENSO file

COrrelations betwee cloud cover and ENSO phase

many similar pieces of code as divergence file

Currently just ERA5 but will work towards using CERES too
"""


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import os
import pandas as pd

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_nino_idx
from obs_scripts.divergence import crop_era5


cz_domain = [-30, 30, 120, -80]


def calc_corr_field(xr_ds, var1='sst', var2='hcc'):
    """
    Calculates the correlation betwen two (already mean subtracted) fields
    in the xr dataset
    """
    proc_df = xr_ds.copy(deep=True)
    
    std_1 = proc_df[var1].std(dim='time')
    std_2 = proc_df[var2].std(dim='time')
    
    proc_df["denom"] = (proc_df[var1] - proc_df[var1].mean()) *\
        (proc_df[var2] - proc_df[var2].mean()) 
    mean_term = proc_df["denom"].mean(dim='time')
    
    corr = mean_term / (std_1 * std_2)
    return corr


def calc_corr_vect(xr_ds, var1, vect, var2='3.4_anom'):
    """
    Calculates the correlation between a field and a vector (ex. Nino 3.4)
    """
    proc_df = xr_ds.copy(deep=True)
    
    std_1 = proc_df[var1].std(dim='time')
    std_2 = np.std(vect[var2])
    
    # Set to all zeros
    proc_df['denom'] = proc_df['hcc'] - proc_df['hcc']
    
    # Since the nio data I have only goes to June
    for n, time in enumerate(proc_df.time[:-3]):
        year_dev = proc_df[{'time':n}][var1] - proc_df[var1].mean()
        year = int(time.dt.year)
        month = int(time.dt.month)
        vect_sel = vect.query(f'year=={year}').query(f'month=={month}')
        proc_df['denom'][{'time':n}] = year_dev *\
            float((vect_sel[var2] - vect[var2].mean()).iloc[0])
    
    mean_term = proc_df["denom"].mean(dim='time')
    corr = mean_term / (std_1 * std_2)
    return corr    


def plot_scalar_simple(data, var, date="2022-12", title='',
                       lims=cz_domain, cbar='HCC (Frac)'):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Not the dictionary filtering version
    """
    era5 = data.sel(time=date)
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if era5.lon.max() > 180:
        era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180))
        era5 = era5.sortby('lon')
    
    # Extract data for plotting
    data = era5[var].data[0]
    lon = era5.lon.values
    lat = era5.lat.values
    
    # Print diagnostics to confirm data ranges
    print(f"Data variable '{var}' - min: {data.min().item():.3f}, max: {data.max().item():.3f}")
    # print(f"Longitude range: {lon.min()} to {lon.max()}")
    # print(f"Latitude range: {lat.min()} to {lat.max()}")

    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    
    # Use pcolormesh for a continuous plot
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), shading='auto', cmap='RdBu_r')
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=0.65)
    cbar.set_label(f'{var} (Frac)')
    
    if lims is not None:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
    plt.show()
    
    
def plot_scalar_field(data, title='',  lims=cz_domain, cbar_lab='HCC (Frac)'):
    """
    Contour plot of a scalar field by proviing the data directly
    """
    era5 = data
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if era5.lon.max() > 180:
        era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180))
        era5 = era5.sortby('lon')
    
    # Extract data for plotting
    data = era5
    lon = era5.lon.values
    lat = era5.lat.values
    
    # Print diagnostics to confirm data ranges
    print(f"Data variable - min: {data.min().item():.3f}, max: {data.max().item():.3f}")
    # print(f"Longitude range: {lon.min()} to {lon.max()}")
    # print(f"Latitude range: {lat.min()} to {lat.max()}")

    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create plot
    # plt.figure()
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    
    # Use pcolormesh for a continuous plot
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), shading='auto', cmap='RdBu_r')
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=0.65)
    cbar.set_label(cbar_lab)
    
    if lims is not None:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
    plt.show()
    
    
def plot_corr(corr_field, title='', lims=cz_domain, cbar_lab='R'):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    """
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if corr_field.lon.max() > 180:
        corr_field = corr_field.assign_coords(lon=(((corr_field.lon + 180) % 360) - 180))
        corr_field = corr_field.sortby('lon')
    
    # Extract data for plotting
    lon = corr_field.lon.values
    lat = corr_field.lat.values
    
    # Print diagnostics to confirm data ranges
    print(f"Corr min: {corr_field.min().item():.3f}, max: {corr_field.max().item():.3f}")
    # print(f"Longitude range: {lon.min()} to {lon.max()}")
    # print(f"Latitude range: {lat.min()} to {lat.max()}")

    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    
    # Use pcolormesh for a continuous plot
    pcm = ax.pcolormesh(lon2d, lat2d, corr_field.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='RdBu_r')
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=0.65)
    cbar.set_label(cbar_lab)
    
    if lims is not None:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
    plt.show()


def domain_sst_anom(era5_anom):
    """
    Calculates the domain sst anomaly in era5 and returns a pandas df
    with year, month, sst_anom (to match the nino formatting)
    """
    sst = era5_anom['sst'].mean(dim=['lat', 'lon']).values
    
    df = {'year': era5_anom.time.dt.year,
          'month': era5_anom.time.dt.month,
          'sst_anom': sst}
    return pd.DataFrame(df)
    

def main():
    era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
    # Crop to region (also renames to lat/lon for conveinence)
    era5_sing = crop_era5(era5_sing, rename=True)

    # Contains nino 3.4 anomaly
    nino_idx = load_nino_idx('misc_data/nino_all.csv')

    # Create time axis
    era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
    # Assign coordinate
    era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
    # Swap coordinate and drop old one
    era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')


    # get the mean year for us
    climatology = era5_sing.groupby('time.month').mean(dim='time')

    era5_anom = era5_sing.copy(deep=True).sel(time=slice("2000-01", "2023-12"))

    years = np.unique(era5_sing.time.dt.year)
    # Anomaly time series
    for n, year in enumerate(years[:-1]):
        # Skip 2024
        year_data = era5_sing.sel(time=slice(f'{year}-01', f'{year}-12'))
        time_axis = year_data.time
        # Reassign time axis so it is consistent with the selected slice
        climatology['time'] = time_axis.data
        climatology = climatology.assign_coords(time=("month",
                                                     climatology.time.data))
        climatology = climatology.swap_dims({"month": "time"})
        
        # Subtract climatology year by year       
        era5_anom[{"time":slice(12 * n, 12 * (n + 1))}] -= climatology   
        
    # Calculate correlations
    
    cloud_class = 'lcc'
    
    # Can set this to 1 if we want to see the vars directly
    weight = era5_sing.mean(dim='time')
    
    plot_scalar_field(weight[cloud_class], title=f'ERA5 {cloud_class.upper()} Climatology',
                      cbar_lab='frac')
    
    # If we don't want to weight the variation
    weight = {cloud_class:1}
    
    corr = calc_corr_field(era5_anom, cloud_class, 'sst')
    plot_corr(corr * weight[cloud_class], cbar_lab='R',
              title=f' Correlation between {cloud_class.upper()} and Local SST Anom')
    
    corr = calc_corr_vect(era5_anom, cloud_class, nino_idx)
    plot_corr(corr * weight[cloud_class], cbar_lab='R',
              title=f' Correlation between {cloud_class.upper()} and Nino 3.4 Anom')

    sst_anom = domain_sst_anom(era5_anom)
    corr = calc_corr_vect(era5_anom, cloud_class, sst_anom, 'sst_anom')
    plot_corr(corr * weight[cloud_class], cbar_lab='R',
              title=f' Correlation Between {cloud_class.upper()} and Mean SST Anom')


if __name__ == '__main__':
    main()
