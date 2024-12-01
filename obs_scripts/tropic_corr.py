# -*- coding: utf-8 -*-
"""
Global-level correlations in FT temperature

Analogous to cloud_corr but I don't crop to a smaller region immediately
"""

import xarray as xr
import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')

from obs_scripts.vis_clouds import load_nino_idx, progress_bar
from obs_scripts.divergence import crop_era5
import obs_scripts.cloud_corr as corr


cz_domain = [-30, 30, 120, -80]


def main():
    # Contains nino 3.4 anomaly
    nino_idx = load_nino_idx('misc_data/nino_all.csv')
    fname = 'era5_reanal/tropics/era5_anom_trop.nc'    
    global era5_anom_trop, inside_trop, outside_trop
    if os.path.exists(fname):
        print('Files found. Loading from netCDF')
        era5_anom_trop = xr.open_dataset(fname)
    else:
        print('Files not found. Calculating from reanalysis..')
        # repeat of cloud_corr, essentially
        era5_sing = xr.open_dataset('era5_reanal/era5_reanal_modern.nc').drop_vars('expver')
        # Remove october extra so the datasets match
        era5_eis = xr.open_dataset('era5_reanal/era5_eis.nc').drop_vars('expver')[{"date":slice(0, -1)}]
        # Crop to region (also renames to lat/lon for conveinence)
        era5_sing = era5_sing.rename({'latitude': 'lat', 'longitude': 'lon'})
        era5_eis = era5_eis.rename({'latitude': 'lat', 'longitude': 'lon'})
        # ceres_syn = crop_era5(ceres_syn, rename=True, coord='360')

        # Create time axis
        era5_sing['time'] = pd.to_datetime(era5_sing.date, format='%Y%m%d')
        era5_eis['time'] = pd.to_datetime(era5_eis.date, format='%Y%m%d')
        # Assign coordinate
        era5_sing = era5_sing.assign_coords(time=("date", era5_sing.time.data))
        era5_eis = era5_eis.assign_coords(time=("date", era5_eis.time.data))
        # Swap coordinate and drop old one
        era5_sing = era5_sing.swap_dims({"date": "time"}).drop_vars('date')
        era5_eis = era5_eis.swap_dims({"date": "time"}).drop_vars('date')

        # Calculate EIS and others and add to single level data
        era5_sing['eis'], era5_sing['theta_700'] = corr.calc_eis(era5_eis)
        era5_sing['t_500'] = era5_eis['t'].sel(pressure_level=500)
        era5_sing['w_700'] = era5_eis['w'].sel(pressure_level=700)
        era5_sing['rh_700'] = era5_eis['r'].sel(pressure_level=700)
        era5_sing['speed'] = np.hypot(era5_sing['u10'], era5_sing['v10'])

        # get the mean year for us
        climatology = era5_sing.groupby('time.month').mean(dim='time')    
        era5_anom = era5_sing.copy(deep=True).sel(time=slice("2000-01", "2023-12"))
        years = np.unique(era5_anom.time.dt.year)
        # Anomaly time series
        for n, year in enumerate(years):
            progress_bar(n, len(years), 'Deseasonalizing...')
            year_data = era5_anom.sel(time=slice(f'{year}-01', f'{year}-12'))
            time_axis = year_data.time
            # Reassign time axis so it is consistent with the selected slice
            climatology['time'] = time_axis.data
            climatology = climatology.assign_coords(time=("month",
                                                         climatology.time.data))
            climatology = climatology.swap_dims({"month": "time"})
            
            # Subtract climatology year by year       
            era5_anom[{"time":slice(12 * n, 12 * (n + 1))}] -= climatology
        
        # DETREND
        print('Detrending...')
        era5_anom = era5_anom.map(lambda da: corr.polyfit_detrend(da, 'time'))
        
        era5_anom.to_netcdf(fname)
        era5_anom_trop = era5_anom
           
        
    overlap_nino = nino_idx.query('2000 <= year <= 2023')
    t_700_globe = corr.domain_anom(era5_anom_trop, 'theta_700')
    corr.plot_combined(overlap_nino['3.4_anom'], t_700_globe['theta_700_anom'], 
                      era5_anom_trop.time, 'Nino 3.4', 'Mean Θ₇₀₀',
                      'Months', 'All Tropics', '', '', sig=0.99)
    
    outside_trop = crop_era5(era5_anom_trop, mode='outside', rename=False)
    t_700_out = corr.domain_anom(outside_trop, 'theta_700')
    corr.plot_combined(overlap_nino['3.4_anom'], t_700_out['theta_700_anom'], 
                      era5_anom_trop.time, 'Nino 3.4', 'Mean Θ₇₀₀',
                      'Months', 'Tropics outside Pacfic', '', '', sig=0.99)
    
    inside_trop = crop_era5(era5_anom_trop, rename=False)
    t_700_in = corr.domain_anom(inside_trop, 'theta_700')
    overlap_nino = nino_idx.query('2000 <= year <= 2023')
    corr.plot_combined(overlap_nino['3.4_anom'], t_700_in['theta_700_anom'], 
                  era5_anom_trop.time, 'Nino 3.4', 'Mean Θ₇₀₀',
                  'Months', 'Tropical Pacific', '', '', sig=0.99)
    
    eof, pc_pac = corr.calc_eof(inside_trop, var='sst', n_pc=4, plot=False)
    pc_pac = corr.rotate_enso_eof(pc_pac)
    
    # Compare to the rotated EOFs
    corr_e = corr.calc_corr_vect(t_700_globe, 'theta_700', pc_pac,
                                 'E', sig=0.99)
    corr.plot_corr(corr_e, cbar_lab='R',
                   title='Correlation Between E Mode and Theta 700')
    
    corr_c = corr.calc_corr_vect(t_700_globe, 'theta_700', pc_pac,
                                 'C', sig=0.99)
    corr.plot_corr(corr_c, cbar_lab='R',
                   title='Correlation Between C Mode and Theta 700')
    
    
if __name__ == '__main__':
    main()    
    