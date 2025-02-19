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
from scipy.signal import correlate, correlation_lags, detrend
from climlab.utils.thermo import EIS
import cartopy.crs as ccrs
import os

# Domains
cz_domain_360 = [-30, 30, 120, -80 + 360]
ep_domain_360 = [-30, -15, 240, 280]
cz_domain_180 = [-30, 30, 120, -80]
ep_domain_180 = [-30, 10, -120, -80]
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
    data = pd.read_csv('misc_data/nino_all.csv', header=0,
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
    
    
def plot_scalar_field(data, title='',  lims=cz_domain_180, cbar_lab='LCC (Frac)'):
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
    
    # Determine the color limits to center around zero
    max_abs = max(abs(data.min().item()), abs(data.max().item()))
    
    # Print diagnostics to confirm data ranges
    # print(f"Data variable - min: {data.min().item():.3f}, max: {data.max().item():.3f}")
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
    pcm = ax.pcolormesh(lon2d, lat2d, data, transform=ccrs.PlateCarree(), 
                        shading='auto', cmap='RdBu_r', vmin=-max_abs, 
                        vmax=max_abs)
    
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
    
    
def plot_corr(corr_field, title='', lims=cz_domain_180, cbar_lab='R',
              shrink=0.65, mode='corr', contour=False, cz_corr=True):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Slope_val determines percentile bounds for slope plot
    """
    # corr_field = corr_field.copy(deep=True)
    # Define central longitude to correctly handle global data
    proj = ccrs.PlateCarree(central_longitude=180)
    
    # Ensure longitude wraps correctly if in 0-360 range
    if corr_field.lon.max() > 180:
        corr_field['lon'] = ((corr_field.lon + 180) % 360) - 180
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
    max_abs = max(abs(corr_field.min().item()), abs(corr_field.max().item()))
    if mode != 'corr':
        quantile = mode.strip('slope_')
        if len(quantile) > 0:
            quantile = float(quantile)
        else:
            quantile = 0.1
        nonan = corr_field.data[~np.isnan(corr_field.data)]
        min_quant, max_quant = np.quantile(nonan, quantile), np.quantile(nonan, 1 - quantile)
        max_abs = max(min_quant, max_quant)

    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    masked_data = np.ma.masked_invalid(corr_field.data)
    
    # Use pcolormesh with centered color limits around zero
    pcm = ax.pcolormesh(lon2d, lat2d, corr_field.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    pcm2 = ax.pcolormesh(lon2d, lat2d, nonsig.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='Greys', alpha=0.1)
    
    if contour:
        contour = ax.contour(lon2d, lat2d, masked_data, levels=np.linspace(-max_abs, max_abs, 8),
                         transform=ccrs.PlateCarree(), colors='black', linewidths=0.7)
        ax.clabel(contour, inline=True, fontsize=8, fmt="%.2f", inline_spacing=5)
    
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
        if cz_corr:
            ax.set_xlim(lims[3] + 20, lims[2] - 20)
        else:
            ax.set_xlim(lims[3], lims[2])
    if lims is not None and len(lims) == 2:
        # Set to tropics still
        ax.set_ylim(lims[0], lims[1])
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
    

def calc_eis(era5_eis, truncate=True):
    """
    Calculates estimated inversion strength of dataarray and returns the same,
    per Wood, 2006 also returns theta_700 and LTS for the sake of it
    
    I adapted code from climlab.utils.thermo to fix some errors with my prev
    code. Still call their module for other things. Just wanted to ensure
    compatibity with xr objects
    """
    if truncate:
        # Remove last month since our single levels data doesn't have that
        era5_eis = era5_eis[{'time':slice(0, len(era5_eis.time) - 1)}]
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
             detrend=False):
    """
    Calculates the first EOF of the SST anomalies across the Pacific.
    Prints explained variance as well.
    
    Also does a linear detrend if asked for
    """    
    # Retains dataset object for now
    era5_anom = era5_anom[[var]].copy()
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
        # Reduce our area to equator as per Takahashi
        era5_anom = era5_anom.sel(lat=slice(-10, 10))
    elif region=='tropics':
        era5_anom = era5_anom.sel(lat=slice(-30, 30))
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