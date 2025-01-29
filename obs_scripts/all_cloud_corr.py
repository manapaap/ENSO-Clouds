# -*- coding: utf-8 -*-
"""
Cloud Correlations for all ERA5 1940-Present

Rather than just the ceres-overlap time of 2000-2023

Very similar to cloud_corr file, although also intended to include
more ideas from ceres_ep, looking at trends averaged in a little SEP box.
I will try to call existing functions rather than re-create them here

We also now have data from 50 N to 50 S rather than just 30N to 30S
"""


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import os


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import load_nino_idx, progress_bar
from obs_scripts.divergence import crop_era5
import obs_scripts.cloud_corr as cloud
from obs_scripts.ceres_ep import plot_regression


# Some domains for convenience
cz_domain = [-30, 30, 120, -80]
# For plotting correlations and scalar
ep_region = [-30, 0, 240, 280]
# Larger pacific
pac_domain = [-50, 50, 120, -60]


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


def isolate_ep(era5_data, ep_domain, var='lcc'):
    """
    Isolates the EP region from larger era5 data for purposes of timeseries
    analysis over averaged quantities
    
    Select single var or get whole set
    """
    era5_ep = era5_data.copy(deep=True)
    lat_bounds = ep_domain[:2][::-1]
    lon_bounds = ep_domain[2:]
    
    era5_ep['lon'] = (era5_ep.lon + 360) % 360
    
    if var:
        era5_ep = era5_ep[var].sel(lat=slice(*lat_bounds), 
                                   lon=slice(*lon_bounds))
    else:
        era5_ep = era5_ep.sel(lat=slice(*lat_bounds), 
                                   lon=slice(*lon_bounds))
    return era5_ep


def main():    
    global era5_data, pc_enso
    file_era5 = 'era5_all/timeseries/era5_anom_all.nc'
    
    if os.path.exists(file_era5):
        print('Files Exist. Loading in data.')
        era5_data = xr.open_dataset(file_era5)
    else: 
        print("Files don't exist. Generating from raw data")
        # Avg contains mean rates, data contains the few things I really care for
        # era5_sing_avg = xr.open_dataset('era5_all/raw/era5_sing_1.nc')
        era5_data = xr.open_dataset('era5_all/raw/era5_sing_0.nc')
        era5_pres = xr.open_dataset('era5_all/raw/era5_pres.nc')
        # Clean up the data
        # era5_sing_avg = era5_sing_avg.drop_vars('expver').rename({'valid_time':
        #                                                      'time'})
        era5_data = era5_data.drop_vars('expver').rename({'valid_time':
                                                              'time'})
        era5_pres = era5_pres.drop_vars('expver').rename({'valid_time':
                                                              'time'})   
        # Add fields we care about from pressure levels
        era5_data['eis'], era5_data['theta_700'] = cloud.calc_eis(era5_pres,
                                                                  truncate=False)
        era5_data['t_500'] = era5_pres['t'].sel(pressure_level=500)
        era5_data['rh_700'] = era5_pres['r'].sel(pressure_level=700)
        era5_data['rh_1000'] = era5_pres['r'].sel(pressure_level=1000)
        era5_data['speed'] = np.hypot(era5_data['u10'], era5_data['v10'])
        
        era5_data = crop_era5(era5_data, rename=True, domain=pac_domain)
        # Mean year
        climatology = era5_data.groupby('time.month').mean(dim='time')
        years = np.unique(era5_data.time.dt.year)
        # Anomaly time series
        for n, year in enumerate(years):
            progress_bar(n, len(years), 'deseasonalizing ERA5...')
            year_data = era5_data.sel(time=slice(f'{year}-01', f'{year}-12'))
            time_axis = year_data.time
            # Reassign time axis so it is consistent with the selected slice
            climatology['time'] = time_axis.data
            climatology = climatology.assign_coords(time=("month",
                                                         climatology.time.data))
            climatology = climatology.swap_dims({"month": "time"})
            # Subtract climatology year by year       
            era5_data[{"time":slice(12 * n, 12 * (n + 1))}] -= climatology   
            
        print('/nSaving to file.../n')
        era5_data.to_netcdf(file_era5)   
        
    # Analysis, but bigger!
    # TODO: analysis using the 700 hpa T data
    
    _, pc_enso = cloud.calc_eof(era5_data, 'sst', n_pc=4,
                                plot=False, region='equator')
    # Just so +ve PC1 is +ve ENSO, as per cloud_corr
    pc_enso['PC1'] *= -1
    pc_enso = cloud.rotate_enso_eof(pc_enso)
    
    # Correlations from before
    corr = cloud.calc_corr_vect(era5_data, 'lcc', pc_enso, 'C')
    cloud.plot_corr(corr, cbar_lab='R', lims=pac_domain,
              title='Correlation Between LCC and C Mode')
    
    _, pc_700 = cloud.calc_eof(era5_data, var='theta_700', n_pc=4, plot=False,
                                region='tropics')
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = cloud.calc_corr_vect(era5_data, 'theta_700', pc_700, 'PC1')
    cloud.plot_corr(corr, cbar_lab='R',
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1')
        
          
if __name__ == "__main__":
    main()
