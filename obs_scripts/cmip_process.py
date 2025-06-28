# -*- coding: utf-8 -*-
"""
Processing CMIP6 models to detrend and try to replicate the reanalysis/
satellite data

Let's try to make this modular...take SST and 
"""


import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import detrend


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
import obs_scripts.shared_funcs as share


def combine_files(folder='CMIP6/NOAA-GFDL/cllcalipso/'):
    """
    Reads and combines all the raw files into a
    single xarray file which can be handled
    more easily
    
    also adds pandas datetime
    """
    files = os.listdir(folder)
    # Empty list to hold files
    loaded = [None for _ in files]
    for n, file in enumerate(files):
        try: 
            loaded[n] = xr.load_dataset(folder + file)
        except:
            # Debug for a few broken files
            print(file)
    output = xr.merge(loaded)
    # Go to single time from time band for convenience
    try:
        # Dumb check since the time format is sometimes already good
        output['time'] = output.indexes['time'].to_datetimeindex()
    except:
        do_nothing = True
    # Set to monthly so days are -01
    output = output.\
        assign_coords(time=output.indexes['time'].to_period('M').to_timestamp())
    
    if 'time_bnds' in list(output.data_vars):
        # Only in GFDL so far
        output = output.drop_vars(['time_bnds', 'lat_bnds', 'lon_bnds'],
                                  errors='ignore')
    elif 'plev_bounds' in list(output.data_vars):
        # This is for IPSL now...
        output = output.drop_vars(['time_bounds', 'plev_bounds', 'plev'],
                                  errors='ignore')
    return output


def deseasonalize(data):
    """
    De-seasonalizes the data by subtracting the mean year from every entry
    done month-by-month to reduce memory overhead. Also does the linear
    detrend
    """
    years = data.time.dt.year
    months = data.time.dt.month
    num = len(years)
    # Calc clim
    clim = data.groupby('time.month').mean(dim='time')
    for n, (year, month) in enumerate(zip(years, months)):
        share.progress_bar(n, num, f'Deseasonalizing...{int(year)}-{int(month)}')
        data[{'time': n}] -= clim.sel(month=month)
    # data = share.polyfit_detrend(data)
    return data


def calc_nino_anom(data, rem_trend=True):
    """
    Calculates the average SST anomaly within the nino 3.4 region
    
    0-360 coords
    """
    region = data.sel(lat=slice(-5, 5),
                      lon=slice(190, 240))
    nino = region.mean(dim=('lat', 'lon'))
    if rem_trend:
        nino = detrend(nino.tos.to_numpy())
    return nino


def main():
    global ipsl_cll, ipsl_sst, ipsl_eof
    start_date = '1983-07-01'
    end_date = '2017-06-02'

    gfdl_cll = combine_files('CMIP6/NOAA-GFDL-CM4/cllcalipso/').\
        sel(time=slice(start_date, end_date))
    gfdl_sst = combine_files('CMIP6/NOAA-GFDL-CM4/sst/').\
        sel(time=slice(start_date, end_date))
        
    gfdl_cll = deseasonalize(gfdl_cll)
    gfdl_sst = deseasonalize(gfdl_sst)
    # restrict sst field to pacific domain
    gfdl_crop = gfdl_sst.sel(lon=slice(120, 300))
    _, gfdl_eof = share.calc_eof(gfdl_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    # as always, fix sign 
    gfdl_eof['PC2'] *= -1
    gfdl_eof = share.rotate_enso_eof(gfdl_eof)
    gfdl_eof['nino_3.4'] = calc_nino_anom(gfdl_sst)
    
    corr = share.calc_corr_vect(gfdl_cll, 'cllcalipso', gfdl_eof, 'C')
    share.plot_corr(corr, cbar_lab='R', lims=share.pac_domain,
              title='Correlation Between GFDL Cll and C')
    # It works! and we fail to see a correlation in the region of intrest
    
    # Let's try this with E3SM
    e3sm_cll = combine_files('CMIP6/E3SM-1-1/cllcalipso/').\
        sel(time=slice(start_date, end_date)).drop_vars(['plev_bnds', 'plev'])
    e3sm_sst = combine_files('CMIP6/E3SM-1-1/sst/').\
        sel(time=slice(start_date, end_date))
        
    e3sm_cll = deseasonalize(e3sm_cll)
    e3sm_sst = deseasonalize(e3sm_sst)
    # restrict sst field to pacific domain
    e3sm_crop = e3sm_sst.sel(lon=slice(120, 300))
    _, e3sm_eof = share.calc_eof(e3sm_crop, 'tos', n_pc=2, exclude_land=False,
                              plot=False, region='equator', detrend=True)
    e3sm_eof['PC2'] *= -1
    e3sm_eof = share.rotate_enso_eof(e3sm_eof)
    e3sm_eof['nino_3.4'] = calc_nino_anom(e3sm_sst)
    
    # try this with IPSL CM6
    ipsl_cll = combine_files('CMIP6/IPSL-CM6A/cllcalipso/').\
        sel(time=slice(start_date, end_date))    
    ipsl_sst = combine_files('CMIP6/IPSL-CM6A/sst_clean/').\
        sel(time=slice(start_date, end_date))  
    
    ipsl_cll = deseasonalize(ipsl_cll)
    ipsl_sst = deseasonalize(ipsl_sst)
    # Fix coords
    ipsl_sst['lon'] = share.convert_longitude(ipsl_sst.lon, to_360=True)
    ipsl_sst = ipsl_sst.sortby('lon')
    ipsl_crop = ipsl_sst.sel(lon=slice(120, 300))
    _, ipsl_eof = share.calc_eof(ipsl_crop, 'tos', n_pc=2, exclude_land=False,
                                 plot=False, region='equator', detrend=True)
    ipsl_eof['PC1'] *= -1
    ipsl_eof['PC2'] *= -1
    ipsl_eof = share.rotate_enso_eof(ipsl_eof)
    ipsl_eof['nino_3.4'] = calc_nino_anom(ipsl_sst)
    # Steps:
    # Combine files for my sanity
    # restrict to ISCCP period
    # remove annual cycle
    # calculate SST PCs and rotate
    # 


if __name__ == '__main__':
    main()

