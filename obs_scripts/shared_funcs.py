# -*- coding: utf-8 -*-
"""
File holding shared functions across all files in obs_scripts folder

Simplifies syntax and prevents circular imports

I finally decided that this must happen
"""


import numpy as np
import xarray as xr
import pandas as pd
import sys
import matplotlib.pyplot as plt
from eofs.xarray import Eof
from scipy.stats import t, linregress
from scipy.signal import correlate, correlation_lags
from climlab.utils.thermo import EIS
import cartopy.crs as ccrs
import os
from scipy.optimize import curve_fit
import scipy.stats as stats
import scipy.signal as signal
import warnings
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import Polygon
from pyproj import Geod
from scipy.ndimage import gaussian_filter


# Domains
cz_domain_360 = [-30, 30, 120, -80 + 360]
ep_domain_360 = [-20, 0, 240, 280]
cz_domain_180 = [-30, 30, 120, -80]
ep_domain_180 = [-20, 0, -120, -80]
# eq ep for PC2 corr
eqp_domain_360 = [-10, 10, 240, 280]
eqp_domain_180 = [-10, 10, -120, -80]
# Larger pacific
pac_domain = [-50, 50, 120, -60]
# NIno 3.4
nino_domain = [-5, 5, -170, -120]


# Formerly in vis_clouds.py
def load_nino_idx(fpath):
    """
    Loads the nino indices and formats it into pandas df with dates
    
    Source:
        https://psl.noaa.gov/data/correlation/nina34.anom.data
        https://psl.noaa.gov/data/timeseries/monthly/NINO34/
    """
    data = pd.read_csv(fpath, header=0,
                       names=['year', 'month', '1_2', '1_2_anom', '3',
                              '3_anom', '4', '4_anom', '3.4', '3.4_anom'])
    
    data['time'] = pd.to_datetime(data['year'].astype(str) + "-" +\
                                  data['month'].astype(str), format='%Y-%m')
    return data


def load_oni_idx(fpath='misc_data/oni_index.txt'):
    """
    Loads ONI index rather than the nino 3.4 index
    """
    oni_df = pd.read_csv(fpath, sep='  ', skiprows=1, 
                         names=['season', 'year', 'oni', 'anom'],
                         engine='python')
    months = oni_df.season.str.slice(0, 3)
    years = oni_df.season.str.slice(4, 10)
    # Fix the weird offset
    oni_df['anom'] = oni_df['oni'].astype(float)
    oni_df['total'] = oni_df['year'].astype(int)
    oni_df['season'] = months
    oni_df['year'] = years.astype(float)
    
    season_to_month = {
        "DJF": "01",  # January
        "JFM": "02",  # February
        "FMA": "03",  # March
        "MAM": "04",  # April
        "AMJ": "05",  # May
        "MJJ": "06",  # June
        "JJA": "07",  # July
        "JAS": "08",  # August
        "ASO": "09",  # September
        "SON": "10",  # October
        "OND": "11",  # November
        "NDJ": "12"   # December
    }
    oni_df['month'] = oni_df['season'].map(season_to_month)
    oni_df['time'] = pd.to_datetime(oni_df['year'].astype(str).str.slice(0, 4) +\
                                    '-' + oni_df['month'].astype(str), format='%Y-%m')
    
    return oni_df


def progress_bar(n, max_val, cus_str=''):
    """
    I love progress bars in long loops
    """
    sys.stdout.write('\033[2K\033[1G')
    print(f'Computing...{100 * (n + 1) / max_val:.2f}% complete ' + cus_str,
          end="\r") 


def is_enso_oni(oni_df, date, cutoff=0.5, out=False):
    """
    determines if given month (as provided by date) is El Nino, La Nina, or
    neutral depending on oni index
    """
    # Convert the date to a datetime object and get the 5-month window
    date = pd.to_datetime(date, format='%Y.%m')
    # date_start = date - pd.DateOffset(months=5)

    # Select the relevant 5-month window
    vals = oni_df.loc[(oni_df['time'] <= date) &\
                      (oni_df['time'] >= date)]
    # Determine ENSO phase based on cutoff value
    if (vals['anom'] >= cutoff).all():
        state = 'El Nino'
    elif (vals['anom'] <= -cutoff).all():
        state = 'La Nina'
    else: 
        state = 'Neutral'    
    # Optionally print the ENSO state
    if out:
        print(state)
    return state


def plot_enso(nino, var='3.4_anom', cutoff=0.4, idx='Nino 3.4'):
    """
    Creates a plot of the nino 3.4 index and cutoffs
    """
    plt.figure()
    plt.plot(nino['time'], nino[var], color='black')
    plt.hlines(cutoff, xmin=nino['time'].iloc[0], xmax=nino['time'].iloc[-1],
               linestyle='dashed', alpha=0.8, color='grey')
    plt.hlines(-cutoff, xmin=nino['time'].iloc[0], xmax=nino['time'].iloc[-1],
               linestyle='dashed', alpha=0.8, color='grey')
    plt.xlabel('Years')
    plt.ylabel(idx + 'anomaly')
    plt.title(idx + 'Index')
    plt.grid()    
    ax = plt.gca()
    ax.fill_between(nino['time'], cutoff, nino[var], 
                    where=nino[var] > cutoff,
                    alpha=0.6, color='red')
    ax.fill_between(nino['time'], -cutoff, nino[var], 
                    where=nino[var] < -0.4,
                    alpha=0.6, color='blue')
    plt.show()


# Formerly in divergence.py
def crop_era5(xr_array, rename=True, coord='180', domain=cz_domain_180,
              mode='inside'):
    """
    Crops to CZ modeldimentions
    
    Returns the region OUTSIDE our box if mode=='outside'
    """
    min_lat, max_lat, east_lon, west_lon = domain
    
    if rename:
        xr_array = xr_array.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    if mode=='inside':
        if coord=='180':
            ds_east = xr_array.sel(lat=slice(max_lat, min_lat), 
                                   lon=slice(east_lon, 180))
            ds_west = xr_array.sel(lat=slice(max_lat, min_lat),
                                   lon=slice(-180, west_lon))
            df = xr.concat([ds_east, ds_west], dim="lon")
            
        elif coord=='360':
            df = xr_array.sel(lat=slice(min_lat, max_lat), 
                              lon=slice(east_lon, 360 + west_lon))
        else:
            print('How do you bork your own code this bad...')
    else:
        if coord == '180':
            # Latitude mask (everything outside min_lat to max_lat)
            lat_mask = ((xr_array['lat'] > max_lat) | (xr_array['lat'] < min_lat))
        
            # Longitude mask (everything outside east_lon to west_lon)
            lon_mask_east = (xr_array['lon'] > east_lon) & (xr_array['lon'] <= 180)
            lon_mask_west = (xr_array['lon'] < west_lon) & (xr_array['lon'] >= -180)
            lon_mask = ~(lon_mask_east | lon_mask_west)  # Invert tropical Pacific region
        
            # Combine the masks
            mask = lat_mask | lon_mask
        
            # Apply the mask
            df = xr_array.where(mask, drop=True)
        else:
            print('Need to implement for 360...')

    return df


def load_composites(directory='era5_reanal/anomalies/', clim=True):
    """
    Loads the composite dictionaries (comp_pres, comp_sing, comp_rad) from NetCDF files.
    Returns the dictionaries with the same structure as in the script.
    """
    comp_pres, comp_sing, comp_rad = {}, {}, {}
    
    composite_dict = {'pres': comp_pres, 'sing': comp_sing, 'rad': comp_rad}
    phases = ['el_nino', 'la_nina', 'neutral']
    if clim:
        phases.append('clim')
    
    for comp_type, comp_dict in composite_dict.items():
        for phase in phases:
            filename = os.path.join(directory, f'{comp_type}_{phase}.nc')
            if os.path.exists(filename):
                comp_dict[phase] = xr.open_dataset(filename)
                print(f'Loaded {phase} composite for {comp_type} from {filename}')
            else:
                print(f'Warning: {filename} does not exist. Skipping load for {comp_type} {phase}.')
                comp_dict[phase] = None
    
    return comp_pres, comp_sing, comp_rad


# Formerly in cloud_corr.py
def calc_corr_field(xr_ds, var1='sst', var2='hcc', sig=0.99, mode='corr'):
    """
    Calculates the correlation betwen two (already mean subtracted) fields
    in the xr dataset
    
    if slope, var1 = slope * var2
    """
    proc_df = xr_ds.copy()
    
    std_1 = proc_df[var1].std(dim='time')
    std_2 = proc_df[var2].std(dim='time')
    
    denom = (proc_df[var1] - proc_df[var1].mean()) *\
        (proc_df[var2] - proc_df[var2].mean()) 
    mean_term = denom.mean(dim='time')
    
    corr = mean_term / (std_1 * std_2)
    
    # Add significance check
    t_stat = corr * np.sqrt((len(xr_ds.time) - 2) / (1 - corr**2))
    # Adjust sig_level for two-tailed test
    adjusted_sig = 1 - (1 - sig) / 2
    t_crit = t.ppf(adjusted_sig, df=len(xr_ds.time) - 2)
    
    # Mask insignificant values
    sig_corr = corr.where(np.abs(t_stat) > t_crit)
    if mode == 'slope':
        # Slope = covariance / variance
        sig_corr /= std_2**2

    return sig_corr


def calc_corr_vect(xr_ds, var1, vect, var2='3.4_anom', sig=0.99, mode='corr'):
    """
    Calculates the correlation between a field and a vector (e.g., Nino 3.4).
    if slope, var1 = slope * var2
    """
    # Convert the vector data to a time-indexed DataArray
    vect['time'] = pd.to_datetime(dict(year=vect['year'], month=vect['month'], day=1))
    vect = vect.set_index('time')
    vect_da = xr.DataArray(vect[var2], coords=[vect.index], dims=['time'])
    # Find the intersection of time periods
    common_times = xr_ds.time.to_index().intersection(vect_da.time.to_index())
    # Align datasets to this common time range
    proc_df = xr_ds.sel(time=common_times)
    vect_da = vect_da.sel(time=common_times)
    # Calculate mean and standard deviation
    std_1 = proc_df[var1].std(dim='time')
    std_2 = vect_da.std(dim='time')
    # Calculate covariance term
    mean_diff_var1 = proc_df[var1] - proc_df[var1].mean(dim='time')
    mean_diff_vect = vect_da - vect_da.mean(dim='time')
    # Calculate covariance and correlation
    mean_term = (mean_diff_var1 * mean_diff_vect).mean(dim='time')
    corr = mean_term / (std_1 * std_2)
    # Add significance check
    t_stat = corr * np.sqrt((len(xr_ds.time) - 2) / (1 - corr**2))
    # Adjust sig_level for two-tailed test
    adjusted_sig = 1 - (1 - sig) / 2
    t_crit = t.ppf(adjusted_sig, df=len(xr_ds.time) - 2)
    # Mask insignificant values
    sig_corr = corr.where(np.abs(t_stat) > t_crit)
    if mode == 'slope':
        # Slope = covariance / variance
        sig_corr /= std_2**2

    return sig_corr


def calc_corr_vect_monthly(xr_ds, var1, vect, var2='3.4_anom', sig=0.99, mode='corr'):
    """
    Calculates the monthly correlation between a field and a vector (e.g., Nino 3.4), 
    with each month treated separately.
    
    Parameters:
    - xr_ds: xarray Dataset containing the field to correlate.
    - var1: str, variable name in xr_ds to correlate.
    - vect: DataFrame containing the vector data with 'year' and 'month' columns and `var2` data.
    - var2: str, variable name in vect for correlation.
    - sig: float, significance level for masking insignificant correlations.
    
    Returns:
    - sig_corr: xarray DataArray with 12 slices (one for each month) of significant correlations.
    """
    # Convert the vector data to a time-indexed DataArray
    vect['time'] = pd.to_datetime(dict(year=vect['year'], month=vect['month'], day=1))
    vect = vect.set_index('time')
    vect_da = xr.DataArray(vect[var2], coords=[vect.index], dims=['time'])
    # Find the intersection of time periods
    common_times = xr_ds.time.to_index().intersection(vect_da.time.to_index())
    # Align datasets to this common time range
    proc_df = xr_ds.sel(time=common_times)
    vect_da = vect_da.sel(time=common_times)
    # Initialize list to collect monthly correlation DataArrays
    monthly_corrs = []

    for month in range(1, 13):  # Loop through each month (1 = Jan, ..., 12 = Dec)
        # Select data for the specific month
        proc_df_month = proc_df.sel(time=proc_df['time.month'] == month)
        vect_da_month = vect_da.sel(time=vect_da['time.month'] == month)
        # Calculate mean and standard deviation for the selected month
        std_1 = proc_df_month[var1].std(dim='time')
        std_2 = vect_da_month.std(dim='time')
        # Calculate covariance term
        mean_diff_var1 = proc_df_month[var1] - proc_df_month[var1].mean(dim='time')
        mean_diff_vect = vect_da_month - vect_da_month.mean(dim='time')
        # Calculate covariance and correlation
        mean_term = (mean_diff_var1 * mean_diff_vect).mean(dim='time')
        corr = mean_term / (std_1 * std_2)
        # Add significance check
        t_stat = corr * np.sqrt((len(proc_df_month.time) - 2) / (1 - corr**2))
        adjusted_sig = 1 - (1 - sig) / 2
        t_crit = t.ppf(adjusted_sig, df=len(proc_df_month.time) - 2)
        # Mask insignificant values
        sig_corr = corr.where(np.abs(t_stat) > t_crit)    
        # Convert to slope if wanted
        if mode == 'slope':
            # Slope = covariance / variance
            sig_corr /= std_1**2
        # Append to list with a new dimension for month
        monthly_corrs.append(sig_corr.expand_dims(dim={'month': [month]}))
    
    # Concatenate along the new 'month' dimension
    sig_corr_all_months = xr.concat(monthly_corrs, dim='month')
    
    return sig_corr_all_months


def plot_scalar_simple(data, var, date="2022-12", title='',
                       lims=cz_domain_180, cbar='HCC (Frac)'):
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
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), 
                        shading='auto', cmap='RdBu_r')
    
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
    
    
def plot_scalar_field(data, title='', lims=pac_domain, cbar_lab='',
                      levels=4, to=''):
    """
    Contour plot of a scalar field by providing the data directly.
    """
    era5 = data.fillna(0).copy()
    proj = ccrs.PlateCarree(central_longitude=180)
    # plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1)
    # Ensure longitude wraps correctly if in 0-360 range
    if era5.lon.max() > 180:
        era5 = era5.assign_coords(lon=(((era5.lon + 180) % 360) - 180))
        era5 = era5.sortby('lon')

    lon = era5.lon.values
    lat = era5.lat.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Improved color normalization
    vmin, vmax = np.percentile(era5.values, [0.5, 99.5])  # Robust scaling
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  

    fig, ax = plt.subplots(figsize=(10, 5), dpi=600,
                                     subplot_kw={'projection': proj})
    ax.set_global()
    ax.set_title(title)

    # pcolormesh plot
    pcm = ax.pcolormesh(lon2d, lat2d, era5, transform=ccrs.PlateCarree(), 
                        shading='nearest', cmap='RdBu_r', norm=norm)

    # Contour overlay
    levels = np.linspace(vmin, vmax, levels)  # Define contour levels
    #contour = ax.contour(lon2d, lat2d, era5, levels=levels, 
    #                     colors='black', linewidths=0.8, 
    #                     transform=ccrs.PlateCarree())
    lon1d = np.asarray(lon2d).reshape(-1)
    lat1d = np.asarray(lat2d).reshape(-1)
    era1d = np.asarray(era5).reshape(-1)
    contour = ax.tricontour(lon1d, lat1d, era1d, levels=levels, 
                        colors='black', linewidths=0.8, 
                      transform=ccrs.PlateCarree())
    ax.clabel(contour, inline=True, fontsize=8)

    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical',
                        pad=0.05, shrink=0.65, format='%02d')
    cbar.set_label(cbar_lab)
    if lims is not None:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3], lims[2])  

    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)      
    plt.show()
    
    
def plot_scalar_subplot(data=[], titles=[], types=[], names=[],
                        lims=pac_domain, cbar_lab=[],
                        levels=[], to=''):
    """
    creates our 3*N subplot of various variables during CP EN, EP EN, and LN
    
    essentially loops over the plot_scalar_field functions, rewritten for subplots
    
    data is in form of a list of dictionaries containing the data for each phase
    
    titles is the title CP/EP EL Nino/La Nina
    
    cbar_lab, and types is the unit and data type corresponding to data
    names contains the variable name within the dict
    """
    num_rows = len(titles)
    num_cols = len(types)
    
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, 
                        dpi=600, subplot_kw={'projection': proj},
                        figsize=(10, 12))  # or adjust as needed)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.7, bottom=0.01)
    # top=0.35
    for row in range(num_rows):
        # =1 for 3 row subplot
        if row < 2:
            cmap = 'RdBu_r'
        else:
            cmap = 'PuOr_r'
        for col, phase in zip(range(num_cols), data[row].keys()):
            if phase == 'mix_nino':
                #  Overwrite for now
                phase = 'nina'
            var = names[row]
            curr_data = data[row][phase][var].fillna(0).copy()
            # Correction to 0-360
            if curr_data.lon.max() > 180:
                curr_data = curr_data.assign_coords(lon=(((curr_data.lon +\
                                                           180) % 360) - 180))
                curr_data = curr_data.sortby('lon')
            # Get lat/lon axis
            lon = curr_data.lon.values
            lat = curr_data.lat.values
            lon2d, lat2d = np.meshgrid(lon, lat)
            # Improved color normalization
            vmin, vmax = np.percentile(curr_data.values, [0.5, 99.5])  # Robust scaling
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) 
            # plot setup   
            axs[row, col].set_global()
            axs[row, col].set_title(titles[row] + ' during ' + types[col],
                                    fontsize='small', pad=5)
            # pcolormesh plot
            pcm = axs[row, col].pcolormesh(lon2d, lat2d, curr_data,
                                           transform=ccrs.PlateCarree(), 
                                shading='nearest', cmap=cmap, norm=norm)

            # Contour overlay
            tiers = np.linspace(vmin, vmax, levels[row])  # Define contour levels
            lon1d = np.asarray(lon2d).reshape(-1)
            lat1d = np.asarray(lat2d).reshape(-1)
            data1d = np.asarray(curr_data).reshape(-1)
            contour = axs[row, col].tricontour(lon1d, lat1d, data1d, 
                                               levels=tiers, 
                                    colors='black', linewidths=0.8, 
                                  transform=ccrs.PlateCarree())
            axs[row, col].clabel(contour, inline=True, fontsize=4,
                     fmt="%.1f", inline_spacing=5)
            # Map flavour
            axs[row, col].coastlines()
            gl = axs[row, col].gridlines(draw_labels=False, dms=True,
                                         alpha=0.5)
            if row == num_rows - 1:
                gl.bottom_labels = True
                gl.xlabel_style = {'size': 8}
            if col == 0:
                gl.left_labels = True
                gl.ylabel_style = {'size': 8}
            # Colorbar
            cbar = plt.colorbar(pcm, ax=axs[row, col], location='right',
                                pad=0.02, shrink=0.65, aspect=20, ticks=tiers)
            # cbar.set_label(cbar_lab[row], labelpad=-20)
            cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()])
            cbar.ax.tick_params(labelsize=8)
            axs[row, col].text(135, 47.5, cbar_lab[row], fontsize=8.5)
            if lims is not None:
                axs[row, col].set_ylim(lims[0], lims[1])
                axs[row, col].set_xlim(lims[3], lims[2])  
    # Savefig and show
    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)      
    plt.show()


    
def plot_corr(corr_field, title='', lims=pac_domain, cbar_lab='R',
              shrink=0.65, levels=5):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Slope_val determines percentile bounds for slope plot
    """
    corr_field = corr_field.fillna(0)
    # corr_field = corr_field.copy(deep=True)
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if corr_field.lon.max() > 180:
        # I AM CHANGING THIS FOR NOW- ROLL BACK IF BROKEN
        new_lons = ((corr_field.lon + 180) % 360) - 180
        corr_field = corr_field.assign_coords(lon=new_lons)
        corr_field = corr_field.sortby('lon')
    # Extract data for plotting
    lon = corr_field.lon.values
    lat = corr_field.lat.values
    
    nonsig = np.zeros(corr_field.shape)
    nonsig[np.isnan(corr_field)] = 1
    
    # Print diagnostics to confirm data ranges
    print(f"Corr min: {corr_field.min().item():.3f}, max: {corr_field.max().item():.3f}")
    
    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    # Determine the color limits to center around zero
    vmin, vmax = np.percentile(corr_field.values, [0.1, 99.9])  # Robust scaling
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    # masked_data = np.ma.masked_invalid(corr_field.data)
    # Use pcolormesh with centered color limits around zero
    pcm = ax.pcolormesh(lon2d, lat2d, corr_field.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='RdBu_r', norm=norm)
    pcm2 = ax.pcolormesh(lon2d, lat2d, nonsig.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='Greys', alpha=0.1)

    levels = np.linspace(vmin, vmax, levels)
    lon1d = np.asarray(lon2d).reshape(-1)
    lat1d = np.asarray(lat2d).reshape(-1)
    corr1d = np.asarray(corr_field).reshape(-1)
    contour = ax.tricontour(lon1d, lat1d, corr1d, levels=levels, 
                        colors='black', linewidths=0.8, 
                      transform=ccrs.PlateCarree())
    #ax.clabel(contour, inline=True, fontsize=4,
    #         fmt="%.1f", inline_spacing=5)
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Add colorbar and label
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=shrink)
    cbar.set_label(cbar_lab)
    
    # Set plot limits if specified
    if lims is not None and len(lims) == 4:
        ax.set_ylim(lims[0], lims[1])
        ax.set_xlim(lims[3], lims[2])
    if lims is not None and len(lims) == 2:
        # Set to tropics still
        ax.set_ylim(lims[0], lims[1])
    plt.show()


def plot_corr_subplot(data, to_corr, vars1, vars2,
                      titles, types, lims=pac_domain, levels=5, to=''):
    """
    Plots a N data * N to_corr subplot correlating each array with a timeseries
    of entries. Plots the pearson correlation for each plot at 99% sig
    
    data and to_corr are lists containing xarray/pandas dataframes containing
    the information we want to correlate
    
    vars1 and vars2 tell us the native variable names within data and to_corr
    these also correspond to titles/types
    """
    num_rows = len(data)
    num_cols = len(to_corr)

    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, 
                        dpi=600, subplot_kw={'projection': proj},
                        figsize=(10, 9))  # or adjust as needed)
    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.8, bottom=0.01)
    # top=0.65
    
    for row in range(num_rows):
        for col, array in enumerate(to_corr):
            corr = calc_corr_vect(data[row], vars1[row], to_corr[col], vars2[col])
            corr_field = corr.fillna(0)
            # Wrapping for ERA5
            if corr_field.lon.max() > 180:
                corr_field['lon'] = ((corr_field.lon + 180) % 360) - 180
                corr_field = corr_field.sortby('lon')
            # Extract data for plotting
            lon = corr_field.lon.values
            lat = corr_field.lat.values
            # Non sig value mask
            nonsig = np.zeros(corr_field.shape)
            nonsig[np.isnan(corr_field)] = 1
            # Create meshgrid if needed
            lon2d, lat2d = np.meshgrid(lon, lat)
            # Determine the color limits to center around zero
            vmin, vmax = np.percentile(corr_field.values, [0.1, 99.9])  # Robust scaling
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            # Begin plotting
            axs[row, col].set_global()
            axs[row, col].set_title(titles[row] + ' and ' + types[col],
                                    fontsize='small')
            # masked_data = np.ma.masked_invalid(corr_field.data)
            # Use pcolormesh with centered color limits around zero
            pcm = axs[row, col].pcolormesh(lon2d, lat2d, corr_field.data,
                                           transform=ccrs.PlateCarree(),
                                shading='auto', cmap='RdBu_r', norm=norm)
            pcm2 = axs[row, col].pcolormesh(lon2d, lat2d, nonsig.data,
                                            transform=ccrs.PlateCarree(),
                                shading='auto', cmap='Greys', alpha=0.1)
            
            tiers = np.linspace(vmin, vmax, levels)
            lon1d = np.asarray(lon2d).reshape(-1)
            lat1d = np.asarray(lat2d).reshape(-1)
            corr1d = np.asarray(corr_field).reshape(-1)
            contour = axs[row, col].tricontour(lon1d, lat1d, corr1d,
                                               levels=tiers, 
                                    colors='black', linewidths=0.8, 
                                  transform=ccrs.PlateCarree())
            axs[row, col].clabel(contour, inline=True, fontsize=4,
                     fmt="%.1f", inline_spacing=5)
            
            # Add coastlines and gridlines
            axs[row, col].coastlines()
            gl = axs[row, col].gridlines(draw_labels=False, dms=True,
                                         alpha=0.5)
            if row == num_rows - 1:
                gl.bottom_labels = True
                gl.xlabel_style = {'size': 8}
            if col == 0:
                gl.left_labels = True
                gl.ylabel_style = {'size': 8}
            
            # Add colorbar and label
            cbar = plt.colorbar(pcm, ax=axs[row, col], location='right',
                                pad=0.02, shrink=0.75, aspect=20, ticks=tiers)
            cbar.ax.set_yticklabels(['{:.1f}'.format(x) for x in cbar.get_ticks()])
            cbar.ax.tick_params(labelsize=8)
            # Set plot limits if specified
            if lims is not None and len(lims) == 4:
                axs[row, col].set_ylim(lims[0], lims[1])
                axs[row, col].set_xlim(lims[3], lims[2])
            if lims is not None and len(lims) == 2:
                # Set to tropics still
                axs[row, col].set_ylim(lims[0], lims[1])
    # Savefig and show
    if to:
        fig.savefig(f'figures\saves\{to}.png', dpi=600,
                    bbox_inches='tight', pad_inches=0)  
    plt.show()
            
            
def domain_anom(era5_anom, var):
    """
    Calculates the domain sst anomaly in era5 and returns a pandas df
    with year, month, sst_anom (to match the nino formatting)
    """
    sst = era5_anom[var].mean(dim=['lat', 'lon'], skipna=True).values
    
    df = {'year': era5_anom.time.dt.year,
          'month': era5_anom.time.dt.month,
          var + '_anom': sst}
    return pd.DataFrame(df)
    

def calc_eis(era5_eis):
    """
    Calculates estimated inversion strength of dataarray and returns the same,
    per Wood, 2006 also returns theta_700 and LTS for the sake of it
    
    I adapted code from climlab.utils.thermo to fix some errors with my prev
    code. Still call their module for other things. Just wanted to ensure
    compatibity with xr objects
    """
    t_700 = era5_eis.sel(pressure_level=700)['t']
    t_1000 = era5_eis.sel(pressure_level=1000)['t']
    # R/cp from wikipedia
    theta_700 = t_700 * (1000 / 700)**(0.286)
    # Get LTS
    LTS = (theta_700 - t_1000)
    # climlab EIS
    eis = EIS(t_1000, t_700)
    return eis, LTS, theta_700


def calc_ectei(era5_eis):
    """
    TODO: Calculates ECTEI as per Kawai, 2017
    
    Assumes we aready have EIS in the dataarray
    """
    ectei = era5_eis['eis'] - 0.23 * 8
    pass

    
def polyfit_detrend(dataarray, dim='time'):
    # Fit a polynomial and subtract the trend
    trend = dataarray.polyfit(dim=dim, deg=1)
    fit = xr.polyval(dataarray[dim], trend.polyfit_coefficients)
    return dataarray - fit


def calc_eof(era5_anom, var, n_pc=1, plot=False, norm=True, region='all',
             detrend=False, exclude_land=False, sst_data=None):
    """
    Calculates the first EOF of the SST anomalies across the Pacific.
    Prints explained variance as well.
    
    Also does a linear detrend if asked for
    
    uses sst mask to remove land values of variables defined over both
    You can now provide sst as a mask separately, but it must be the same
    shape 
    """    
    # Retains dataset object for now
    era5_anom = era5_anom[[var]].copy()
    if exclude_land and sst_data is None:
        sst_data = era5_anom['sst']
        era5_anom[var] = era5_anom[var].where(sst_data.notnull())
    if detrend:
        era5_anom = era5_anom.map(lambda da: polyfit_detrend(da, 'time'))
    era5_anom = era5_anom[var]
    # Adjust longitude coordinates to range 0-360 if necessary
    if era5_anom.lon.min() < 0:
        era5_anom = era5_anom.assign_coords(lon=(era5_anom.lon % 360))
        era5_anom = era5_anom.sortby('lon')
    # Sort by latitude as well if necessary
    era5_anom = era5_anom.sortby('lat')
    if region=='equator':
        # Reduce our area to Pacific equator as per Takahashi
        era5_anom = era5_anom.sel(lat=slice(-10, 10))
    elif region=='tropics':
        era5_anom = era5_anom.sel(lat=slice(-30, 30))
    elif region=='NP':
        era5_anom = era5_anom.sel(lat=slice(20, 90))
    # Initialize EOF solver
    solver = Eof(era5_anom)
    # Calculate the first EOF
    # eofs = solver.eofs(neofs=n_pc)
    # Calculate the first principal component
    pcs = solver.pcs(npcs=n_pc)
    # Print explained variance for modes
    var_frac = solver.varianceFraction(neigs=n_pc).data
    errors = solver.northTest(neigs=n_pc, vfscaled=True)
    for n, var_amt in enumerate(var_frac):
        print(f'PC{n + 1} Variance Explained: {float(100 * var_amt):.2f} %')
    if plot:
        plt.figure()
        plt.grid()
        plt.xlabel('Principal Component')
        plt.errorbar(np.arange(1, n_pc + 1), 100 * var_frac,
                     yerr=100 * errors, capsize=20)
        plt.ylabel('Variance Explained (%)')
        plt.title(f'PCs of {var.upper()}')
    # Turn pc into pandas df to match used format
    pcs_df = pd.DataFrame({'year': era5_anom.time.dt.year.data,
                        'month': era5_anom.time.dt.month.data})
    for n in range(n_pc):
        pcs_df['PC' + str(n + 1)] = pcs.sel(mode=n).data
        if norm:
            pcs_df['PC' + str(n + 1)] /= np.std(pcs.sel(mode=n).data)
    # simpler time index for plotting
    pcs_df['simp_time'] = pcs_df.year + ((pcs_df.month - 1) / 12)
    return solver, pcs_df

    
def plot_combined(series1, series2, time_axis, name1, name2, dt='months', 
                  title='', sig=0.99, norm=True, cutoff=12):
    """
    Creates a set of subplots with time series, 1D correlation, and lag plots.
    
    series1, series2: Data series to analyze and plot.
    time_axis: Time data for the time series plot.
    name1, name2: Names of the series for labeling.
    dt: Time unit (e.g., 'days' or 'months') for labeling the lag plot.
    title: Title for the figure.
    """
    series1 = series1.copy()
    series2 = series2.copy()
    # Calculate linear regression for scatter plot
    reg = linregress(series1, series2)
    
    # Calculate lags and correlation for lag plot
    lags = correlation_lags(len(series1), len(series2), 'same')
    correl = correlate(series1 / np.std(series1), series2 / np.std(series2),
                       'same')
    max_lag = lags[abs(correl) == abs(correl).max()]
    correl /= len(series1)
    # Significance thresholds for R
    n = len(lags)
    # Since we lose degrees of freedom with more lag
    n_vect = n - abs(lags)
    # We need to now adjust this by the autocorrelation
    auto_1 = correlate(series1 / np.std(series1), series1 / np.std(series1),
                       'same') / len(series1)
    auto_2 = correlate(series2 / np.std(series2), series2 / np.std(series2),
                       'same') / len(series1)
    auto_1 = auto_1[lags == 1]
    auto_2 = auto_2[lags == 1]
    n_vect = n_vect * (1 - auto_1 * auto_2) / (1 + auto_1 * auto_2)
    # Adjust sig_level for two-tailed test
    adjusted_sig = 1 - (1 - sig) / 2
    t_crit = t.ppf(adjusted_sig, df=n_vect - 2)
    correl_min = -t_crit / np.sqrt(n_vect - 2 + t_crit**2)
    correl_max = t_crit / np.sqrt(n_vect - 2 + t_crit**2)
    # Let's now isolate it to +- 2 years range max
    correl = correl[abs(lags) < cutoff]
    correl_min = correl_min[abs(lags) < cutoff]
    correl_max = correl_max[abs(lags) < cutoff]
    lags = lags[abs(lags) < cutoff]
    # Normalization
    if norm:
        series1_og = series1.copy()
        series2_og = series2.copy()
        series1 /= np.std(series1)
        series2 /= np.std(series2)
    # Set up the subplots    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8),
                            gridspec_kw={'height_ratios': [0.8, 1]})
    # fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=0.35)    
    # Top plot (spans two columns): Time series
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(time_axis, series1, label=name1)
    ax_top.plot(time_axis, series2, label=name2)
    ax_top.set_xlabel('Time')
    ax_top.set_ylabel('Mag')
    ax_top.grid()
    ax_top.legend()
    ax_top.set_title(title)
    # Turn off the underlying axes
    axs[0, 0].get_yaxis().set_visible(False)
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)
    # Bottom left plot: Scatter plot with regression line
    axs[1, 0].scatter(series1_og, series2_og, color='black', alpha=0.7, s=17.5)
    axs[1, 0].plot(series1_og, reg.slope * series1 + reg.intercept, 
                   linestyle='dashed', color='red', linewidth=3, zorder=10)
    axs[1, 0].set_xlabel(name1)
    axs[1, 0].set_ylabel(name2)
    axs[1, 0].grid()
    axs[1, 0].set_title(f'Scatter Plot (R² = {reg.rvalue**2:.3f})')
    # Bottom right plot: Lag plot    
    axs[1, 1].plot(lags, correl, label='Pearson Correlation', zorder=10, color='black')
    axs[1, 1].plot(lags, correl**2, label='Variance Explained', zorder=9, 
                   color='black', linestyle='dashed')
    axs[1, 1].fill_between(lags, y1=correl_min, y2=correl_max, alpha=0.25,
                           color='red', label=f'Not Significant at {sig}', zorder=1,
                           linewidth=3.5)
    axs[1, 1].plot(lags, correl_min, linestyle='dashed', alpha=0.4, color='red',
                   linewidth=1.5, zorder=2)
    axs[1, 1].plot(lags, correl_max, linestyle='dashed', alpha=0.4, color='red',
                   linewidth=1.5, zorder=2)
    axs[1, 1].set_xlabel(f'Lag ({dt})')
    axs[1, 1].set_ylabel('R')
    axs[1, 1].grid()
    # Axis lims for clarity
    max_height = np.max(correl) + 0.05
    min_height = np.min([np.min(correl) - 0.05, 0])
    axs[1, 1].set_ylim(min_height, max_height)
    if max_lag[0] < 0:
        diff = 'Leads'
    else:
        diff = 'Lags'
    axs[1, 1].set_title(f'{name1} {diff} {name2} by {abs(max_lag[0])} {dt}')
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and labels
    plt.show() 
    
    return lags, correl


def rotate_enso_eof(pc_enso):
    """
    Calculates the rotated "C" and "E" modes of variation as per
    Takahashi, Ken, et al. "ENSO regimes: Reinterpreting the canonical 
    and Modoki El Niño." Geophysical research letters 38.10 (2011).
    """
    normed_pc1 = pc_enso['PC1'] / np.std(pc_enso['PC1'])
    normed_pc2 = pc_enso['PC2'] / np.std(pc_enso['PC2'])
    
    pc_enso['C'] = (normed_pc1 + normed_pc2) / np.sqrt(2)
    pc_enso['E'] = (normed_pc1 - normed_pc2) / np.sqrt(2)
    
    return pc_enso


# Formerly in ceres_ep
def plot_regression(arr1, arr2, xlabel='', ylabel='', title=''):
    """
    Plots the regression of arr1 and arr2, along with the best fit line
    """
    
    reg = linregress(arr1, arr2)
    x_range = np.linspace(arr1.min(), arr1.max())
    
    plt.figure()
    plt.scatter(arr1, arr2, label='Observations')
    plt.grid()
    plt.plot(x_range, reg.slope * x_range + reg.intercept, label=\
             f'R²={reg.rvalue**2:.3f}'+ f'\np = {reg.pvalue:.3f}' +\
                 f'\nm={reg.slope:.3f}',
             linestyle='dashed', color='red')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()  


def ep_cp_ensos(pc_enso):
    """
    Uses the E and C indices to define E vs C dominant nature
    of ENSO events
    Returns a df containing year, enso state, fe and fc, calculated as
    fe = E / (E + C) and complementary
    """
    fe = pc_enso['E'] / (np.abs(pc_enso['C']) + np.abs(pc_enso['E']))
    fc = pc_enso['C'] / (np.abs(pc_enso['C']) + np.abs(pc_enso['E']))
    
    pattern = pd.DataFrame({'year': pc_enso.year,
                            'month': pc_enso.month,
                            'fe': fe, 'fc': fc})
    
    return pattern


# Formerly in all_cloud_corr.py
def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = signal.butter(order, normalCutoff, btype='low', analog=False)  # Use analog=False for a digital filter
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y


def red_AR1(f, autocorr, A):
    """
    Red noise spectrum for AR1 process from
    https://en.wikipedia.org/wiki/Autoregressive_model#AR(1)
    """
    rs = A * (1.0 - autocorr**2) /(1. -(2.0 * autocorr * np.cos(f * 2.0 * np.pi))\
                                   + autocorr**2)
    return rs


def red_brownian(f, A):
    """
    Red noise fit for Brownian motion
    """
    return A / (2 * np.pi * f**2)


def red_OU(f, A, sigma):
    """
    red noise fit for Ornstein-Uhlenbeck process from:
    https://arxiv.org/pdf/2212.03566
    """
    return sigma / (A**2 + (2 * np.pi * f)**2)


def plot_psd(array, nperseg=256, sig=0.99, cutoff=1, 
             period=1, nfft=256, name='SST', how='AR1'):
    """
    Plots the psd and red noise null hypothesis to check for significant peaks
    under "cutoff"
    
    Returns relevant parameters to reconstruct the figure
    
    WE can now choose what red noise we want!
    """    
    f, Pxx = signal.welch(array , fs=1/period, nperseg=nperseg, nfft=nfft)
    Pxx /= Pxx.mean()
    # cut off the high frequency bs
    Pxx = Pxx[f < cutoff]
    f = f[f < cutoff]
    # Get lag-1 autocorr to fit the red noise
    corr = signal.correlate(array / np.std(array), array / np.std(array),
                            mode='full')[array.shape[0]:] / len(array)
    corr_1 = float(corr[1])
    # Fit the red noise function
    if how =='AR1':
        red_params, red_covar = curve_fit(red_AR1, f, Pxx, p0=(corr_1, 1))
        red_fitted = red_AR1(f, *red_params)
    elif how == 'brownian':
        red_params, red_covar = curve_fit(red_brownian, f, Pxx, p0=(1))
        red_fitted = red_brownian(f, *red_params)
    elif how =='OU':
        red_params, red_covar = curve_fit(red_OU, f, Pxx, p0=(1, 1))
        red_fitted = red_OU(f, *red_params)
    
    # Calculate f-value
    n_df = 2 * len(array) / nperseg
    m_df = len(array) / 2
    f_stat = stats.f.ppf(sig, n_df, m_df)
    # Plot the PSD
    # This way to extract the peak assumes only one sig peak but that is OK
    # Reworkng this
    plt.figure()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        num_peaks = len(f[Pxx > (f_stat * red_fitted)])
        peaks = f[Pxx > (f_stat * red_fitted)]
    
    # Rescale to per-yera by multiply by 12
    plt.semilogx(f * 12, Pxx, label=f'{name}')
    plt.vlines([(3)**-1, (7)**-1], min(Pxx), max(f_stat * red_fitted),
                  label='ENSO Freqs', color='red', linestyle='dashed')
    plt.vlines((10)**-1, min(Pxx), max(f_stat * red_fitted),
                  label='Decadal ENSO', color='darkred', linestyle='dashdot')
    plt.semilogx(f * 12, red_fitted, label='Fitted Red Noise')
    plt.semilogx(f * 12, f_stat * red_fitted, label=f'{sig} Significance')
    plt.grid()
    plt.ylabel('Power')
    plt.xlabel('Frequency (Cycles / year)')
    plt.legend()
    plt.title(f'Power Spectrum of {name}')
    
    spectral_params = {'f': f,
                       'Pxx': Pxx,
                       'red_fit': red_fitted,
                       'f_stat': f_stat,
                       'peaks': peaks}
    return spectral_params


def plot_csd(arr1, arr2, nperseg=256, period=1, nfft=512, unit='months',
             var1='', var2='', plot='log'):
    """
    Plots the cross spectral density of two variables; includes
    the phase lag plot.
    
    Parameters:
    - arr1, arr2: Input time series (numpy arrays)
    - nperseg: Number of data points per segment for Welch's method
    - period: Sampling period (e.g., 1 for yearly, etc.)
    - nfft: Number of FFT points
    - unit: Unit of the time lag (default: 'months')
    """
    arr1 = arr1.copy()
    arr2 = arr2.copy()
    # Remove mean and normalize by standard deviation
    arr1 = (arr1 - np.mean(arr1)) / np.std(arr1)
    arr2 = (arr2 - np.mean(arr2)) / np.std(arr2)
    # Remove linear trend
    arr1 = signal.detrend(arr1)
    arr2 = signal.detrend(arr2)
    # Compute the power spectral densities
    f, Pxx = signal.welch(arr1, fs=1/period, nperseg=nperseg, nfft=nfft)
    f, Pyy = signal.welch(arr2, fs=1/period, nperseg=nperseg, nfft=nfft)
    # Compute the cross power spectral density
    f, Pxy = signal.csd(arr1, arr2, fs=1/period, nperseg=nperseg, nfft=nfft)
    # Compute magnitude (coherence-like measure)
    mag = np.abs(Pxy) / np.sqrt(Pxx * Pyy)  # Fixed normalization
    # Compute phase (in radians)
    phase = np.angle(Pxy)  # Fixed phase calculation
    # Compute time lag (handling f = 0 case to avoid division by zero)
    lag = np.zeros_like(phase)
    nonzero_f = f > 0  # Avoid division by zero at f=0
    lag[nonzero_f] = phase[nonzero_f] / (2 * np.pi * f[nonzero_f])
    # Plot results
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    if var1 and var2:
        fig.suptitle(f'{var1} vs {var2}')
    plt.tight_layout()

    if plot=='log':
        axs[0].semilogx(f, mag, label="Magnitude")
    else:
        axs[0].plot(f, mag, label="Magnitude")
    axs[0].set_ylabel('Magnitude')
    axs[0].set_title('Cross Spectral Density')
    axs[0].grid()
    axs[0].vlines([(12*3)**-1, (12*7)**-1], min(mag), max(mag),
                  label='ENSO Freqs', color='red', linestyle='dashed')
    axs[0].legend()
    
    if plot=='log':
        axs[1].semilogx(f, lag, label=f"Lag ({unit})")
    else:
        axs[1].plot(f, lag, label=f"Lag ({unit})")
    axs[1].set_ylabel(f'Lag ({unit})')
    axs[1].set_xlabel('Frequency')
    axs[1].vlines([(12*3)**-1, (12*7)**-1], min(lag), max(lag),
                  label='ENSO Freqs', color='red', linestyle='dashed')
    axs[1].grid()
    axs[1].legend()
    
    plt.show()
    
    
def isolate_ep_era5(era5_data, var='lcc', domain=ep_domain_360):
    """
    Isolates the EP region from larger era5 data for purposes of timeseries
    analysis over averaged quantities
    
    Select single var or get whole set
    """
    era5_ep = era5_data.copy(deep=True)
    lat_bounds = domain[:2][::-1]
    lon_bounds = domain[2:]
    era5_ep['lon'] = (era5_ep.lon + 360) % 360
    if var:
        era5_ep = era5_ep[var].sel(lat=slice(*lat_bounds), 
                                   lon=slice(*lon_bounds))
    else:
        era5_ep = era5_ep.sel(lat=slice(*lat_bounds), 
                                   lon=slice(*lon_bounds))
    return era5_ep.mean(dim=['lat', 'lon'])


def isolate_ep_isccp(isccp_anom, var, domain=ep_domain_360):
    """
    Returns mean of variable within isccp_anom in eastern pacific
    """
    data = isccp_anom.sel(lat=slice(*domain[:2]),
                          lon=slice(*domain[2:]))
    return data[var].mean(dim=['lat', 'lon'])


def convert_longitude(lon, to_360=True):
    """
    Convert longitude between -180 to 180 and 0 to 360.

    Parameters:
        lon (float or array-like): Longitude value(s) to convert.
        to_360 (bool): If True, convert from [-180, 180] to [0, 360].
                       If False, convert from [0, 360] to [-180, 180].

    Returns:
        Converted longitude value(s).
    """
    if to_360:
        return (lon + 360) % 360  # Convert -180 to 180 -> 0 to 360
    else:
        return ((lon + 180) % 360) - 180  # Convert 0 to 360 -> -180 to 180


def plot_pcs(pc_enso, one='PC1', two='PC2', title=''):
    """
    Plots PC1 vs. PC2 scatter, along with axes for C and E. Tries to fit a 
    polynomial curve to the data as per
    https://www.nature.com/articles/s41586-018-0776-9
    
    Using only Nov-Jan months
    """
    enso_peak = pc_enso.query('month <= 1 or month >= 11')
    
    fit = np.polyfit(enso_peak[one], enso_peak[two], 2)
    
    x = np.linspace(enso_peak[one].min(), enso_peak[one].max(),
                    num=len(enso_peak[one]))
    x2 = np.linspace(pc_enso[one].min(), pc_enso[one].max(),
                    num=len(pc_enso[one]))
    poly = np.polyval(fit, x)
    
    plt.figure(figsize=(5, 5))
    plt.scatter(pc_enso[one], pc_enso[two], alpha=0.5, color='grey',
                label='All Year', zorder=15, s=1)
    plt.scatter(enso_peak[one], enso_peak[two], color='black',
                label='NDJ', zorder=20, s=1.2, alpha=0.8)
    plt.plot(x2, x2, color='darkred', linewidth=0.5, 
             label='C')
    plt.plot(x2, -x2, color='darkblue', linewidth=0.5,
             label='E')
    plt.plot(x, poly, label='Poly Fit', zorder=17)
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()
    

def calc_area(domain):
    """
    Calculates area in m2 for a rectangle defined by the coordinate
    domains at the start of this file
    
    coords shouold be in 0-360
    """
    lat1, lat2, lon1, lon2 = domain
    # Change to coordinates
    coords = [(lon1, lat2), (lon1, lat1),
              (lon2, lat1), (lon2, lat2)]
    # Create a polygon and give it georeference
    poly = Polygon(coords)
    geod = Geod(ellps="WGS84")
    # Calc and return area
    area, _ = geod.geometry_area_perimeter(poly)
    return abs(area)
    
    
def smooth_data(era5_field, sigma=3):
    """
    Gaussian smooths a given field and returns it. Wraps around to prevent
    artifact at 180 deg
    """
    n_wrap = int(3 * sigma)
    wrapped = xr.concat(
        [era5_field.isel(lon=slice(-n_wrap, None)),
         era5_field,
         era5_field.isel(lon=slice(0, n_wrap))],
        dim="lon")
    smoothed_wrapped = xr.apply_ufunc(
        gaussian_filter, wrapped,
        input_core_dims=[["lat", "lon"]],
        output_core_dims=[["lat", "lon"]],
        kwargs={"sigma": sigma},
        vectorize=True)

    # Crop back to original longitude range
    smoothed = smoothed_wrapped.isel(lon=slice(n_wrap, -n_wrap))
    smoothed = smoothed.assign_coords(lon=era5_field.lon)
    return smoothed


def plot_enso_season(data, start_year, field, title='', cbar_lab='',
                      levels=4, to=''):
    """
    Plots the NDJF average of a certain field, starting at start_year
    """
    date1 = f'{start_year}-11-01'
    date2 = f'{start_year+1}-02-28'
    season = data[field].sel(time=slice(date1, date2))
    season = season.mean(dim='time')
    
    plot_scalar_field(season, lims=pac_domain, title=title, cbar_lab=cbar_lab,
                      levels=levels, to=to)
    
    
    
