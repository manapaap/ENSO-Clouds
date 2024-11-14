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
from scipy.signal import correlate, correlation_lags


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_nino_idx
from obs_scripts.divergence import crop_era5


cz_domain = [-30, 30, 120, -80]


def calc_corr_field(xr_ds, var1='sst', var2='hcc', sig=0.95):
    """
    Calculates the correlation betwen two (already mean subtracted) fields
    in the xr dataset
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

    return sig_corr


def calc_corr_vect(xr_ds, var1, vect, var2='3.4_anom', sig=0.95):
    """
    Calculates the correlation between a field and a vector (e.g., Nino 3.4).
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

    return sig_corr
  


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
    
    # Create meshgrid if needed
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Determine the color limits to center around zero
    max_abs = max(abs(corr_field.min().item()), abs(corr_field.max().item()))

    # Create plot
    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=(10, 5))
    ax.set_global()
    ax.set_title(title)
    
    # Use pcolormesh with centered color limits around zero
    pcm = ax.pcolormesh(lon2d, lat2d, corr_field.data, transform=ccrs.PlateCarree(),
                        shading='auto', cmap='RdBu_r', vmin=-max_abs, vmax=max_abs)
    
    # Add coastlines and gridlines
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Add colorbar and label
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', pad=0.05,
                        shrink=0.65)
    cbar.set_label(cbar_lab)
    
    # Set plot limits if specified
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


def calc_enso_eof(era5_anom, var, n_pc=1):
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
    eofs = solver.eofs(neofs=1)

    # Calculate the first principal component
    pcs = solver.pcs(npcs=1)
    
    # Print explained variance for the first mode
    variance_fractions = solver.varianceFraction(neigs=1)
    print(f'Frac Variance Explained: {100 * float(variance_fractions):.2f} %')
    
    # Turn pc into pandas df to match used format
    pcs = pd.DataFrame({'year': era5_anom.time.dt.year.data,
                        'month': era5_anom.time.dt.month.data,
                        'PC1': pcs.data[:, 0]})
    
    return eofs, pcs

    
def plot_combined(series1, series2, time_axis, name1, name2, dt, title,
                  var1, var2):
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
    
    # Set up the subplots    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [0.8, 1]})
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(hspace=0.35)    
    # Top plot (spans two columns): Time series
    ax_top = fig.add_subplot(2, 1, 1)
    ax_top.plot(time_axis, series1, label=name1)
    ax_top.plot(time_axis, series2, label=name2)
    ax_top.set_xlabel('Time')
    ax_top.set_ylabel('PC1')
    ax_top.grid()
    ax_top.legend()
    ax_top.set_title('Time Series')
    
    # Turn off the underlying axes
    axs[0, 0].get_yaxis().set_visible(False)
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 1].get_xaxis().set_visible(False)
    axs[0, 1].get_yaxis().set_visible(False)
    
    # Bottom left plot: Scatter plot with regression line
    axs[1, 0].scatter(series1, series2)
    axs[1, 0].plot(series1, reg.slope * series1 + reg.intercept, 
                   linestyle='dashed', color='red')
    axs[1, 0].set_xlabel(name1 + f' ({var1}% variance)')
    axs[1, 0].set_ylabel(name2 + f' ({var2}% variance)')
    axs[1, 0].grid()
    axs[1, 0].set_title(f'Scatter Plot (R² = {reg.rvalue**2:.3f})')
    
    # Bottom right plot: Lag plot
    axs[1, 1].plot(lags, correl)
    axs[1, 1].set_xlabel(f'Lag ({dt})')
    axs[1, 1].set_ylabel('Correlation (R)')
    axs[1, 1].grid()
    axs[1, 1].set_title(f'{name1} Leads {name2} by {-max_lag[0]} {dt}')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title and labels
    plt.show() 

 
def main():
    global era5_anom
    era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
    era5_eis = xr.open_dataset('era5_reanal/era5_lts.nc').drop_vars('expver')
    # Crop to region (also renames to lat/lon for conveinence)
    era5_sing = crop_era5(era5_sing, rename=True)
    era5_eis = crop_era5(era5_eis, rename=True)

    # Contains nino 3.4 anomaly
    nino_idx = load_nino_idx('misc_data/nino_all.csv')

    # Create time axis
    era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
    era5_eis['time'] = pd.to_datetime(era5_eis.date, format='%Y%m%d')
    # Assign coordinate
    era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
    era5_eis = era5_eis.assign_coords(time=("date", era5_eis.time.data))
    # Swap coordinate and drop old one
    era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')
    era5_eis = era5_eis.swap_dims({"date": "time"}).drop_vars('date')

    # Calculate EIS and add to single level data
    era5_sing['eis'], era5_sing['theta_700'] = calc_eis(era5_eis)

    # get the mean year for us
    climatology = era5_sing.groupby('time.month').mean(dim='time')
    # Analysis variable
    cloud_class = 'lcc'
    
    # plot_scalar_field(climatology.sel(month=12)[cloud_class],
    #                   title=f'ERA5 December {cloud_class.upper()} Climatology',
    #                   cbar_lab='frac')

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
    sig = 0.99
    # Can set this to 1 if we want to see the vars directly
    if False:
        weight = era5_sing.mean(dim='time')
        
        # Let's skip plotting low clouds constantly until we need the
        plot_scalar_field(weight[cloud_class], title=f'ERA5 {cloud_class.upper()} Climatology',
                          cbar_lab='frac')
        
        corr = calc_corr_field(era5_anom, cloud_class, 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title=f' Correlation between {cloud_class.upper()} Anom and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, cloud_class, nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title=f' Correlation between {cloud_class.upper()} Anom and Nino 3.4 Anom')
            
        # This isn't all that intrestig so we will not consider it. 
        # sst_anom = domain_sst_anom(era5_anom)
    
        # EIS correlations
        corr = calc_corr_field(era5_anom, 'eis', 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between EIS Anom and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, 'eis', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between EIS Anom and Nino 3.4 Anom')
        
        # FT Temp Correlation
        corr = calc_corr_vect(era5_anom, 'theta_700', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between Θ₇₀₀ Anom and Nino 3.4 Anom')
        
        corr = calc_corr_field(era5_anom, 'theta_700', 'sst', sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between 700 HPa Potential Temp. and Local SST Anom')
        
        corr = calc_corr_vect(era5_anom, 'sst', nino_idx, sig=sig)
        plot_corr(corr, cbar_lab='R',
                  title='Correlation Between SST Anom and Nino 3.4 Anom')
        
    # LEt's corrlate this with the ENSO PC1 since that captures the spacial
    # variance of this better
    eof, pc_700 = calc_enso_eof(era5_anom, var='theta_700')
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = calc_corr_vect(era5_anom, 'theta_700', pc_700, 'PC1', sig=sig)
    plot_corr(corr, cbar_lab='R',
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1 (30.5% Variance)')
    
    # Let's correlate the ENSO PC to the Theta_700 pc
    eof, pc_enso = calc_enso_eof(era5_anom, var='sst')

    plot_combined(pc_enso['PC1'], pc_700['PC1'], era5_anom.time, 'SST PC1', 'Θ₇₀₀ PC1',
                  'Months', '', 37.5, 30.6)
    

if __name__ == '__main__':
    main()
