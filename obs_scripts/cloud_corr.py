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
from eofs.xarray import Eof
from scipy.stats import t, linregress
from scipy.signal import correlate, correlation_lags, detrend
import gc


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_nino_idx, progress_bar
from obs_scripts.divergence import crop_era5


cz_domain = [-30, 30, 120, -80]


def calc_corr_field(xr_ds, var1='sst', var2='hcc', sig=0.95, mode='corr'):
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


def calc_corr_vect(xr_ds, var1, vect, var2='3.4_anom', sig=0.95, mode='corr'):
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


def calc_corr_vect_monthly(xr_ds, var1, vect, var2='3.4_anom', sig=0.95, mode='corr'):
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
    
    
def plot_scalar_field(data, title='',  lims=cz_domain, cbar_lab='LCC (Frac)'):
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
    
    
def plot_corr(corr_field, title='', lims=cz_domain, cbar_lab='R',
              shrink=0.65, mode='corr', contour=False):
    """
    Contour plot of a scalar field (e.g., hcc) across the globe or a specified region.
    
    Slope_val determines percentile bounds for slope plot
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
        ax.set_xlim(lims[3] + 20, lims[2] - 20)
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
    

def calc_eis(era5_eis):
    """
    Calculates estimated inversion strength of dataarray and returns the same
    
    also returns theta_700
    """
    # Remove last month since our single levels data doesn't have that
    era5_eis = era5_eis[{'time':slice(0, len(era5_eis.time) - 1)}]
    t_700 = era5_eis.sel(pressure_level=700)['t']
    t_1000 = era5_eis.sel(pressure_level=1000)['t']
    
    # R/cp from wikipedia
    theta_700 = t_700 * (1000 / 700)**(0.286)
    # Since theta = T at surface reference pressure
    return (theta_700 - t_1000), theta_700


def calc_eof(era5_anom, var, n_pc=1, plot=False, norm=True):
    """
    Calculates the first EOF of the SST anomalies across the Pacific.
    Prints explained variance as well.
    """
    era5_anom = era5_anom[var]
    # Adjust longitude coordinates to range 0-360 if necessary
    if era5_anom.lon.min() < 0:
        era5_anom = era5_anom.assign_coords(lon=(era5_anom.lon % 360))
        era5_anom = era5_anom.sortby('lon')
    # Sort by latitude as well if necessary
    era5_anom = era5_anom.sortby('lat')
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

    
def plot_combined(series1, series2, time_axis, name1, name2, dt, title,
                  var1, var2, sig=0.99, norm=True):
    """
    Creates a set of subplots with time series, 1D correlation, and lag plots.
    
    series1, series2: Data series to analyze and plot.
    time_axis: Time data for the time series plot.
    name1, name2: Names of the series for labeling.
    dt: Time unit (e.g., 'days' or 'months') for labeling the lag plot.
    title: Title for the figure.
    """
    # Calculate linear regression for scatter plot
    reg = linregress(series1, series2)
    
    # Calculate lags and correlation for lag plot
    lags = correlation_lags(len(series1), len(series2), 'same')
    correl = correlate(series1 / np.std(series1), series2 / np.std(series2),
                       'same')
    max_lag = lags[correl == correl.max()]
    correl /= len(series1)
    # Significance thresholds for R
    n = len(lags)
    # Since we lose degrees of freedom with more lag
    n_vect = n - abs(lags)
    # Adjust sig_level for two-tailed test
    adjusted_sig = 1 - (1 - sig) / 2
    t_crit = t.ppf(adjusted_sig, df=n_vect - 2)
    correl_min = -t_crit / np.sqrt(n_vect - 2 + t_crit**2)
    correl_max = t_crit / np.sqrt(n_vect - 2 + t_crit**2)
    # Normalization
    if norm:
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
    axs[1, 0].scatter(series1, series2, color='black', alpha=0.7, s=17.5)
    axs[1, 0].plot(series1, reg.slope * series1 + reg.intercept, 
                   linestyle='dashed', color='red', linewidth=3, zorder=10)
    if var1:
        axs[1, 0].set_xlabel(name1 + f' ({var1}% variance)')
        axs[1, 0].set_ylabel(name2 + f' ({var2}% variance)')
    else:
        axs[1, 0].set_xlabel(name1)
        axs[1, 0].set_ylabel(name2)
    axs[1, 0].grid()
    axs[1, 0].set_title(f'Scatter Plot (R² = {reg.rvalue**2:.3f})')
    
    # Bottom right plot: Lag plot
    axs[1, 1].plot(lags, correl, label='Pearson Correlation', zorder=10, color='black')
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
    axs[1, 1].set_title(f'{name1} Leads {name2} by {-max_lag[0]} {dt}')
    axs[1, 1].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and labels
    plt.show() 


def polyfit_detrend(dataarray, dim='time'):
    # Fit a polynomial and subtract the trend
    trend = dataarray.polyfit(dim=dim, deg=1)
    fit = xr.polyval(dataarray[dim], trend.polyfit_coefficients)
    return dataarray - fit


def load_ceres_syn():
    """
    Loads ceres SYN data from file, combining the 2010-2012 and 2023-2024 files
    into one dataarray
    """
    part1 = xr.open_dataset('ceres_data/ceres_syn_200003-201012.nc')
    part2 = xr.open_dataset('ceres_data/ceres_syn_201301-2024.nc')
    
    return xr.concat([part1, part2], dim='time')


def transform_coord(xr_ds, mode='360 Start'):
    """
    Transforms coordinates from -180 to 180 to 0-360 or vice-versa.
    
    Parameters:
    - xr_ds: xarray Dataset with a 'lon' coordinate.
    - mode: string, either '360 Start' (for -180 to 180 -> 0 to 360) or 
            '-180 to 180' (for 0 to 360 -> -180 to 180).
    
    Returns:
    - Transformed xarray Dataset.
    """
    lon = xr_ds.lon
    if mode == '360 Start':
        # Convert -180 to 180 to 0 to 360
        lon = (lon % 360)  # Wraps into 0 to 360 range
        data = xr_ds.assign_coords(lon=lon)
        data = data.sortby('lon')
        return data
    elif mode == '-180 to 180':
        # Convert 0 to 360 to -180 to 180
        lon = (((lon + 180) % 360) - 180)  # Wraps into -180 to 180 range
        data = xr_ds.assign_coords(lon=lon)
        data = data.sortby('lon')
        return data
    else:
        raise ValueError("Invalid mode. Use '360 Start' or '-180 to 180'")   


def calc_div(era5_sing, era5_pres):
    """
    Adds divergence from pressure levels data to the single levels data
    mean across 1000-850 and just at 950
    """
    mean_d = era5_pres['d'].sel(pressure_level=slice(1000, 850))
    mean_d = mean_d.mean(dim='pressure_level')
    era5_sing['d_mean'] = mean_d
    era5_sing['d_950'] = era5_pres['d'].sel(pressure_level=950)
    
    return era5_sing


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
    
 
def main():
    # Contains nino 3.4 anomaly
    nino_idx = load_nino_idx('misc_data/nino_all.csv')
    nino_idx['3.4_anom'] = detrend(nino_idx['3.4_anom'])
    # Create a lagged index by arbitrary number of months
    nino_idx['3.4_anom_lag'] = nino_idx['3.4_anom'].shift(2)
    
    global era5_anom, ceres_anom, ebaf_anom   
    file_era5 = 'era5_reanal/timeseries/era5_anom.nc'
    file_ceres = 'era5_reanal/timeseries/ceres_syn.nc'
    file_clim = 'era5_reanal/timeseries/climate.nc'
    file_ebaf = 'era5_reanal/timeseries/ceres_ebaf.nc'
    
    files = [file_era5, file_ceres, file_clim, file_ebaf]
    if all([os.path.exists(file) for file in files]):
        print('Files Exist. Loading in data.')
        era5_anom = xr.open_dataset(file_era5)
        ceres_anom = xr.open_dataset(file_ceres)
        era5_sing = xr.open_dataset(file_clim)
        ebaf_anom = xr.open_dataset(file_ebaf)
    else: 
        print('Files not found. Calculating from reanalysis..')
        era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
        # Remove october extra so the datasets match
        era5_pres = xr.open_dataset('era5_reanal/era5_reanal_pres.nc').drop_vars('expver')[{"date":slice(0, -1)}]
        era5_eis = xr.open_dataset('era5_reanal/era5_eis.nc').drop_vars('expver')[{"date":slice(0, -1)}]
        ceres_syn = load_ceres_syn()
        ceres_ebaf = xr.load_dataset('ceres_data/ceres_ebaf_all.nc')
        # Crop to region (also renames to lat/lon for conveinence)
        era5_sing = crop_era5(era5_sing, rename=True)
        era5_eis = crop_era5(era5_eis, rename=True)
        era5_pres = crop_era5(era5_pres, rename=True)
        # ceres_syn = crop_era5(ceres_syn, rename=True, coord='360')
        # Calculate divergence
        era5_sing = calc_div(era5_sing, era5_pres)
       
        # Create time axis
        era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
        era5_eis['time'] = pd.to_datetime(era5_eis.date, format='%Y%m%d')
        # Assign coordinate
        era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
        era5_eis = era5_eis.assign_coords(time=("date", era5_eis.time.data))
        # Swap coordinate and drop old one
        era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')
        era5_eis = era5_eis.swap_dims({"date": "time"}).drop_vars('date')
        
        # Fix CERES dates (since they number on 15th rather than 1st)
        new_time = pd.to_datetime(ceres_syn.time.to_index()).to_period('M').to_timestamp()
        ceres_syn = ceres_syn.assign_coords(time=new_time)
        new_time = pd.to_datetime(ceres_ebaf.time.to_index()).to_period('M').to_timestamp()
        ceres_ebaf = ceres_ebaf.assign_coords(time=new_time)
        
        # Calculate EIS and others and add to single level data
        era5_sing['eis'], era5_sing['theta_700'] = calc_eis(era5_eis)
        era5_sing['t_500'] = era5_eis['t'].sel(pressure_level=500)
        era5_sing['w_700'] = era5_eis['w'].sel(pressure_level=700)
        era5_sing['rh_700'] = era5_eis['r'].sel(pressure_level=700)
        era5_sing['speed'] = np.hypot(era5_sing['u10'], era5_sing['v10'])
        era5_sing['pv_700'] = era5_eis['pv'].sel(pressure_level=700)
        era5_sing['pv_500'] = era5_eis['pv'].sel(pressure_level=500)
    
        # get the mean year for us
        climatology = era5_sing.groupby('time.month').mean(dim='time')
        clim_cer_syn = ceres_syn.groupby('time.month').mean(dim='time')
        clim_ebaf = ceres_ebaf.groupby('time.month').mean(dim='time')
            
        era5_anom = era5_sing.copy(deep=True).sel(time=slice("2000-01", "2023-12"))
        # CERES first starts 2000-03, so we will skip to 2001
        ceres_anom = ceres_syn.copy(deep=True).sel(time=slice("2001-01", "2023-12"))   
        ebaf_anom = ceres_ebaf.copy(deep=True).sel(time=slice('2001-01', '2023-12'))
    
        years = np.unique(era5_anom.time.dt.year)
        # Anomaly time series
        for n, year in enumerate(years):
            progress_bar(n, len(years), 'deseasonalizing ERA5...')
            year_data = era5_anom.sel(time=slice(f'{year}-01', f'{year}-12'))
            time_axis = year_data.time
            # Reassign time axis so it is consistent with the selected slice
            climatology['time'] = time_axis.data
            climatology = climatology.assign_coords(time=("month",
                                                         climatology.time.data))
            climatology = climatology.swap_dims({"month": "time"})
            # Subtract climatology year by year       
            era5_anom[{"time":slice(12 * n, 12 * (n + 1))}] -= climatology   
        # Perform the same on ceres
        years = np.unique(ceres_anom.time.dt.year)
        for n, year in enumerate(years): 
            progress_bar(n, len(years), 'deseasonalizing CERES SYN...')
            year_data = ceres_anom.sel(time=slice(f'{year}-01', f'{year}-12'))
            time_axis = year_data.time
            # Reassign time axis so it is consistent with the selected slice
            clim_cer_syn['time'] = time_axis.data
            clim_cer_syn = clim_cer_syn.assign_coords(time=("month",
                                                         clim_cer_syn.time.data))
            clim_cer_syn = clim_cer_syn.swap_dims({"month": "time"})
            # Subtract climatology year by year       
            ceres_anom[{"time":slice(12 * n, 12 * (n + 1))}] -= clim_cer_syn  
            
        years = np.unique(ebaf_anom.time.dt.year)
        for n, year in enumerate(years): 
            progress_bar(n, len(years), 'deseasonalizing CERES EBAF...')
            year_data = ebaf_anom.sel(time=slice(f'{year}-01', f'{year}-12'))
            time_axis = year_data.time
            # Reassign time axis so it is consistent with the selected slice
            clim_ebaf['time'] = time_axis.data
            clim_ebaf = clim_ebaf.assign_coords(time=("month",
                                                      clim_ebaf.time.data))
            clim_ebaf = clim_ebaf.swap_dims({"month": "time"})
            # Subtract climatology year by year       
            ebaf_anom[{"time":slice(12 * n, 12 * (n + 1))}] -= clim_ebaf  
            
        # Detrend all
        print('Detrending...')
        era5_anom = era5_anom.map(lambda da: polyfit_detrend(da, 'time'))
        ceres_anom = ceres_anom.map(lambda da: polyfit_detrend(da, 'time')) 
        ebaf_anom = ebaf_anom.map(lambda da: polyfit_detrend(da, 'time')) 
        
        # Cloud radiative effect?
        ceres_anom['swcre_surf'] = ceres_anom['adj_atmos_sw_up_clr_surface_mon'] -\
            ceres_anom['adj_atmos_sw_up_all_surface_mon']
        ceres_anom['toa_cre_sw'] = ceres_anom.data_vars['toa_sw_clr_mon'] -\
            ceres_anom.data_vars['toa_sw_all_mon']
        # Surface?
        ebaf_anom['clr_sfc_net'] = ebaf_anom['sfc_cre_net_tot_mon'] +\
            ebaf_anom['sfc_net_tot_all_mon']
        # Column heating
        ebaf_anom['atms_net_all'] = ebaf_anom['toa_net_all_mon'] -\
            ebaf_anom['sfc_net_tot_all_mon']
        
        era5_anom.to_netcdf(file_era5)
        ceres_anom.to_netcdf(file_ceres)
        era5_sing.to_netcdf(file_clim)
        ebaf_anom.to_netcdf(file_ebaf)
    
    # ANALYSIS TIME!!!
    climatology = era5_sing.groupby('time.month').mean(dim='time')
    # Analysis variable
    cloud_class = 'lcc'
    cloud_class_cer = 'cldarea_low_mon' # Same difference

    plot_scalar_field(climatology[cloud_class].sel(month=12) - climatology[cloud_class].mean(dim='month'),
                       title=f'ERA5 December Anomalous {cloud_class.upper()}',
                       cbar_lab='frac')
    # Calculate correlations    
    sig = 0.99
    # Can set this to 1 if we want to see the vars directly
    if True:        
        # Let's skip plotting low clouds constantly until we need the        
        corr = calc_corr_field(era5_anom, cloud_class, 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title=f' Correlation between {cloud_class.upper()} Anom and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, cloud_class, nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title=f' Correlation between {cloud_class.upper()} Anom and Nino 3.4 Anom')
        
        corr = calc_corr_vect(era5_anom, cloud_class, nino_idx, '3.4_anom_lag', 
                              sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title=f' Correlation between {cloud_class.upper()} Anom and 2-Month lag Nino 3.4 Anom')
        
        # Does this replicate with CERES?
        corr = calc_corr_vect(ceres_anom, cloud_class_cer, nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between CERES LCC and Nino 3.4 Anom')
        
        corr = calc_corr_vect(ceres_anom, cloud_class_cer, nino_idx, '3.4_anom_lag', 
                              sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between CERES LCC Anom and 2-Month lag Nino 3.4 Anom')
            
        # This isn't all that intrestig so we will not consider it. 
        # sst_anom = domain_sst_anom(era5_anom)
    
        # EIS correlations
        corr = calc_corr_field(era5_anom, 'eis', 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between EIS Anom and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, 'eis', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between EIS Anom and Nino 3.4 Anom')
        
        corr = calc_corr_vect(era5_anom, 'eis', nino_idx, '3.4_anom_lag',
                              sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between EIS Anom and 2-Month lag Nino 3.4 Anom')
        
        # FT Temp Correlation
        corr = calc_corr_vect(era5_anom, 'theta_700', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between Θ₇₀₀ Anom and Nino 3.4 Anom')
        
        corr = calc_corr_field(era5_anom, 'theta_700', 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between Θ₇₀₀ Anom and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, 'theta_700', nino_idx)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between Θ₇₀₀ Anom Nino 3.4 Anom')
        
        corr = calc_corr_vect(era5_anom, 'sst', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between SST Anom and Nino 3.4 Anom')
        
        # SW Rad correlation
        corr = calc_corr_vect(ceres_anom, 'toa_sw_all_mon', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between CERES TOA SW RAD and Nino 3.4 Anom')
        
    # LEt's corrlate this with the ENSO PC1 since that captures the spacial
    # variance of this better
    eof, pc_700 = calc_eof(era5_anom, var='theta_700', n_pc=4, plot=False)
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = calc_corr_vect(era5_anom, 'theta_700', pc_700, 'PC1', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1 (24.65% Variance)')
    
    # Let's correlate the ENSO PC to the Theta_700 pc
    eof, pc_enso = calc_eof(era5_anom, var='sst', n_pc=4, plot=False)

    plot_combined(pc_enso['PC1'], pc_700['PC1'],
                  era5_anom.time, 'SST PC1', 'Θ₇₀₀ PC1',
                  'Months', '', 38.71, 24.66, sig=0.99)
    
    plot_combined(pc_enso['PC1'], pc_enso['PC2'],
                  era5_anom.time, 'SST PC1', 'SST PC2',
                  'Months', '', 38.71, 9.06, sig=0.99)
    # This is quite unusual, so let's check for domain mean Theta_700
    # Continued in tropic_corr.py...
    corr = calc_corr_vect(era5_anom, 'sst', pc_enso, 'PC2', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between SST PC2 and SST (9.06% Variance)')
    
    # This matches the pattern of more low clouds in SEP and less in NEP
    corr = calc_corr_vect(era5_anom, 'lcc', pc_enso, 'PC2', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between SST PC2 and LCC (9.06% Variance)')
    
    # Alert! Rotated EOFS!
    pc_enso = rotate_enso_eof(pc_enso)
    
    corr = calc_corr_vect(era5_anom, 'lcc', pc_enso, 'C', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between LCC and C Mode')
    corr = calc_corr_vect(era5_anom, 'lcc', pc_enso, 'E', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between LCC and E Mode')
    # These very neatly explain the spatial pattern of cloud cover changes
    
    overlap_nino = nino_idx.query('2000 <= year <= 2023')
    plot_combined(overlap_nino['3.4_anom'], pc_enso['PC1'],
                  era5_anom.time, '3.4 Anom', 'PC1',
                  'Months', '', 0, 0, sig=0.99)

    plot_combined(overlap_nino['3.4_anom'], pc_enso['C'],
                  era5_anom.time, '3.4 Anom', 'C',
                  'Months', '', 0, 0, sig=0.99)

    plot_combined(overlap_nino['3.4_anom'], pc_enso['E'],
                  era5_anom.time, '3.4 Anom', 'E',
                  'Months', '', 0, 0, sig=0.99)
    # This explains the lagged free troposphere response compared to Nino 3.4!
    
if __name__ == '__main__':
    main()
    # Force garbage collection
    gc.collect()
