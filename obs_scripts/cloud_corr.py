# -*- coding: utf-8 -*-
"""
Cloud Correlations and ENSO file

COrrelations betwee cloud cover and ENSO phase

many similar pieces of code as divergence file

Currently just ERA5 but will work towards using CERES too
"""


import numpy as np
import xarray as xr
import os
import pandas as pd
from scipy.signal import detrend

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
import obs_scripts.shared_funcs as share


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
  
 
def main():
    # Contains nino 3.4 anomaly
    nino_idx = share.load_nino_idx('misc_data/nino_all.csv')
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
        era5_sing = share.crop_era5(era5_sing, rename=True)
        era5_eis = share.crop_era5(era5_eis, rename=True)
        era5_pres = share.crop_era5(era5_pres, rename=True)
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
        era5_sing['eis'], era5_sing['lts'], era5_sing['theta_700'] = share.calc_eis(era5_eis)
        era5_sing['t_500'] = era5_eis['t'].sel(pressure_level=500)
        era5_sing['w_700'] = era5_eis['w'].sel(pressure_level=700)
        era5_sing['rh_700'] = era5_eis['r'].sel(pressure_level=700)
        era5_sing['rh_1000'] = era5_eis['r'].sel(pressure_level=1000)
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
            share.progress_bar(n, len(years), 'deseasonalizing ERA5...')
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
            share.progress_bar(n, len(years), 'deseasonalizing CERES SYN...')
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
            share.progress_bar(n, len(years), 'deseasonalizing CERES EBAF...')
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
        era5_anom = era5_anom.map(lambda da: share.polyfit_detrend(da, 'time'))
        ceres_anom = ceres_anom.map(lambda da: share.polyfit_detrend(da, 'time')) 
        ebaf_anom = ebaf_anom.map(lambda da: share.polyfit_detrend(da, 'time')) 
        
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

    share.plot_scalar_field(climatology[cloud_class].sel(month=12) - climatology[cloud_class].mean(dim='month'),
                       title=f'ERA5 December Anomalous {cloud_class.upper()}',
                       cbar_lab='frac')
        
    # LEt's corrlate this with the ENSO PC1 since that captures the spacial
    # variance of this better
    eof, pc_700 = share.calc_eof(era5_anom, var='theta_700', n_pc=1, plot=False)
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = share.calc_corr_vect(era5_anom, 'theta_700', pc_700, 'PC1')
    share.plot_corr(corr, cbar_lab='R',
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1 (24.65% Variance)')
    
    # Let's correlate the ENSO PC to the Theta_700 pc
    eof, pc_enso = share.calc_eof(era5_anom, var='sst', n_pc=4, plot=False,
                            region='equator')

    # This is quite unusual, so let's check for domain mean Theta_700
    # Continued in tropic_corr.py...
    
    # Alert! Rotated EOFS!
    pc_enso = share.rotate_enso_eof(pc_enso)
    # Alert! garbage sign conventions!
    ebaf_anom[['toa_lw_all_mon', 'toa_sw_all_mon']] *= -1
    
    corr = share.calc_corr_vect(era5_anom, 'sst', pc_enso, 'E')
    share.plot_corr(corr, cbar_lab='R',
              title='Correlation Between SST and E Mode')
    corr = share.calc_corr_vect(era5_anom, 'sst', pc_enso, 'C')
    share.plot_corr(corr, cbar_lab='R',
              title='Correlation Between SST and C Mode')    
    
    corr = share.calc_corr_vect(era5_anom, 'lcc', pc_enso, 'C')
    share.plot_corr(corr, cbar_lab='R',
              title='Correlation Between LCC and C Mode')
    corr = share.calc_corr_vect(era5_anom, 'lcc', pc_enso, 'E')
    share.plot_corr(corr, cbar_lab='R',
              title='Correlation Between LCC and E Mode')
    # These very neatly explain the spatial pattern of cloud cover changes
    
    overlap_nino = nino_idx.query('2000 <= year <= 2023')
    share.plot_combined(overlap_nino['3.4_anom'], pc_enso['PC1'],
                  era5_anom.time, '3.4 Anom', 'PC1',
                  'Months', '', sig=0.99)

    share.plot_combined(overlap_nino['3.4_anom'], pc_enso['C'],
                  era5_anom.time, '3.4 Anom', 'C',
                  'Months', '', sig=0.99)

    share.plot_combined(overlap_nino['3.4_anom'], pc_enso['E'],
                  era5_anom.time, '3.4 Anom', 'E',
                  'Months', '', sig=0.99)
    # This explains the lagged free troposphere response compared to Nino 3.4!
    
    # Save ENSO-PCs to be used at-will
    pc_enso.to_csv('misc_data/enso_pcs.csv', index=False)
    
if __name__ == '__main__':
    main()
