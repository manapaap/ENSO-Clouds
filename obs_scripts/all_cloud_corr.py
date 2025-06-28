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
import numpy as np
import xarray as xr
import os
import scipy.signal as signal
import warnings
import pandas as pd


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
import obs_scripts.shared_funcs as share


def calc_oni(era5_data):
    """
    Calculates the oceanic Nino index for our data from 1940-present
    """
    nino_data = era5_data['sst'].sel(lat=slice(*share.nino_domain[:2][::-1]), 
                                     lon=slice(*share.nino_domain[2:]))
    nino_data = nino_data.mean(dim=['lat', 'lon'])
    
    # Need to create the base periods
    starts = np.arange(1936, 1992, 5)
    ends = np.arange(1965, 2022, 5)
    means = []
    
    for start, end in zip(starts, ends):
        period_data = nino_data.sel(time=slice(f'{start}-01-01', 
                                               f'{end}-12-02'))
        means.append(float(period_data.mean()))
      
    N = 0
    for n, time in enumerate(nino_data.time):
        if time.dt.year > ends[N] - 10 and N != len(means) - 1:
            N += 1
        nino_data[{'time': n}] -= means[N]
    
    # Let's put this in a dataframe, as usual
    calc_nino = pd.DataFrame({'year': nino_data.time.dt.year,
                              'month': nino_data.time.dt.month,
                              'time': nino_data.time,
                              '3.4_anom': nino_data.data})
    # Get oceanic nino as 3-mon; handle edges and ensure mean is centered
    oni = calc_nino['3.4_anom'].rolling(window=3).mean()
    calc_nino['oni'] = 0.0
    calc_nino['oni'][1:-1] = oni[2:]
    calc_nino['oni'][0] = np.mean(calc_nino['3.4_anom'][:2])
    calc_nino['oni'].iloc[-1] = np.mean(calc_nino['3.4_anom'][-2:])
    
    return calc_nino    


def noaa_detrend(arr1, time=10, how='mean'):
    """
    Uses the noaa-style detrending to return a detrended variable. This
    is nice since it makes fewer assumptions on the actual shape of the trend

    how can either be by removing the mean, or by removing the linear trend

    arr1 assumed to be an xarray dataarray
    """
    arr1 = arr1.copy()
    start = arr1.time.dt.year[0]
    end = arr1.time.dt.year[-1]
    arr1['simp_time'] = arr1.time.dt.year + ((arr1.time.dt.month - 1) / 12)

    # We will lump the 2020-2029 period with 2010-2019, making 2010-2024
    starts = np.arange(start, end - time, time)
    ends = np.arange(start + 9, end + 9 - time, time)
    # Loop over these ranges of years    
    fits = []
    for start, end in zip(starts, ends):
        period_data = arr1.sel(time=slice(f'{start}-01-01', 
                                              f'{end}-12-02'))
        if how == 'mean':
            fits.append(float(period_data.mean()))
        elif how == 'lm':
            print('NOT IMPLEMENTED YET')

    N = 0
    for n, time in enumerate(arr1.time):
        if time.dt.year > ends[N] and N != len(fits) - 1:
            N += 1
        arr1[{'time': n}] -= fits[N]

    return arr1


def is_enso_monthly(nino_df, cutoff=0.5):
    """
    Assigns enso state to a new column of a pandas df containing the
    oni index and other variables of intrest
    """
    nino_df['enso_state'] = ''
    for n, oni in enumerate(nino_df['oni']):
        if oni >= cutoff:
            nino_df['enso_state'].iloc[n] = 'El Nino'
        elif oni <= -cutoff:
            nino_df['enso_state'].iloc[n] = 'La Nina'
        else:
            nino_df['enso_state'].iloc[n] = 'Neutral'
    return nino_df


def polynomial_detrend(data, time_axis, degree=3, plot=False):
    """
    detrends the data using a third degree
    polynomial; assumes you know what you're
    doing
    
    if plot, plots the fit on the original data
    """
    fit = np.polynomial.polynomial.Polynomial.fit(time_axis, data,
                                                  degree)
    detr = data - fit(time_axis)
    if plot:
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(time_axis, data, label='original')
        axs[0].plot(time_axis, fit(time_axis), label='fit')
        axs[0].legend()
        axs[0].grid()
        
        # linreg = linregress(time_axis, detr)
        axs[1].plot(time_axis, detr)
        axs[1].grid()
    
    return detr


def cold_adv(era5_data):
    """
    Calculates cold air advection using u10/v10 and SST gradients
    
    Shouold be accurate for monthly averaging purposes. 
    """
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0
    # Compute lat/lon in radians
    lat = era5_data.latitude * deg_to_rad
    lon = era5_data.longitude * deg_to_rad
    # Midpoint differences for centered derivatives
    dlat = lat.diff('latitude')
    dlon = lon.diff('longitude')
    # Grid spacing (at midpoints)
    dy = R_earth * dlat  # [m]
    dx = R_earth * np.cos(lat) * dlon  # [m] (no need for lat_mid; use original lat)
    # Compute gradients at CENTER points (using midpoints)
    dsst_dy = era5_data['sst'].differentiate('latitude') / dy
    dsst_dx = era5_data['sst'].differentiate('longitude') / dx
    # Interpolate u10/v10 to SST gradient grid (midpoints)
    u_mid = era5_data['u10'].interp(latitude=dsst_dy.latitude, 
                                    longitude=dsst_dx.longitude)
    v_mid = era5_data['v10'].interp(latitude=dsst_dy.latitude, 
                                    longitude=dsst_dx.longitude)
    # Compute advection
    cold_adv = -(u_mid * dsst_dx + v_mid * dsst_dy)
    return cold_adv


def cold_adv_periodic(era5_data):
    """
    Calculates cold air advection using u10/v10 and SST gradients,
    using periodic longitude derivatives to avoid 180° artifacts.
    """
    R_earth = 6.371e6
    deg_to_rad = np.pi / 180.0

    lat = era5_data['lat'] * deg_to_rad
    lon = era5_data['lon'] * deg_to_rad

    # Assume regular lat/lon spacing
    dlat = float(lat[1] - lat[0])  # radians
    dlon = float(lon[1] - lon[0])  # radians

    # Grid spacing
    dy = R_earth * dlat  # constant
    dx = R_earth * np.cos(lat) * dlon  # varies with lat, shape (lat,)

    # -- Latitude derivative (standard centered diff) --
    dsst_dy = era5_data['sst'].differentiate('lat') / dy  # shape: (time, lat, lon)

    # -- Longitude derivative: periodic central diff --
    sst = era5_data['sst']
    lon_dim = 'lon'
    lat_dim = 'lat'

    sst_roll_p1 = sst.roll({lon_dim: -1}, roll_coords=False)
    sst_roll_m1 = sst.roll({lon_dim: 1}, roll_coords=False)

    # dx is (lat,) so we need to broadcast to match (lat, lon)
    dx_broadcast = dx.broadcast_like(sst.isel({lon_dim: 0}))  # shape (lat, lon)
    dsst_dx = (sst_roll_p1 - sst_roll_m1) / (2 * dx_broadcast)

    # -- Interpolate wind to match SST gradients --
    u_mid = era5_data['u10'].interp(lat=dsst_dy.lat, lon=dsst_dx.lon)
    v_mid = era5_data['v10'].interp(lat=dsst_dy.lat, lon=dsst_dx.lon)

    # -- Compute cold air advection --
    cold_adv = -(u_mid * dsst_dx + v_mid * dsst_dy)

    return cold_adv
    
    
def main():    
    global era5_data, pc_enso, lcc_anom, nino_data, era5_flux
    file_era5 = 'era5_all/timeseries/era5_anom_all.nc'
    file_flux = 'era5_all/timeseries/era5_flux_all.nc'
    
    if os.path.exists(file_era5) and os.path.exists(file_flux):
        print('Files Exist. Loading in data.')
        era5_data = xr.open_dataset(file_era5)
        era5_flux = xr.open_dataset(file_flux)
    else: 
        print("Files don't exist. Generating from raw data")
        # Avg contains mean rates, data contains the few things I really care for
        # era5_data_avg = xr.open_dataset('era5_all/raw/era5_data_1.nc')
        era5_data = xr.open_dataset('era5_all/raw/era5_sing_new_0.nc').drop_vars('number')
        era5_flux = xr.open_dataset('era5_all/raw/era5_sing_new_1.nc')
        era5_pres = xr.open_dataset('era5_all/raw/era5_pres_new.nc')
        # Clean up the data
        # era5_data_avg = era5_data_avg.drop_vars('expver').rename({'valid_time':
        #                                                      'time'})
        era5_data = era5_data.drop_vars('expver').rename({'valid_time':
                                                              'time'})
        era5_flux = era5_flux.drop_vars('expver').rename({'valid_time':
                                                              'time'})
        era5_pres = era5_pres.drop_vars('expver').rename({'valid_time':
                                                              'time'})   
        # Add fields we care about from pressure levels
        era5_data['eis'], era5_data['lts'], era5_data['theta_700'] = share.calc_eis(era5_pres)
        era5_data['t_500'] = era5_pres['t'].sel(pressure_level=500)
        era5_data['w_700'] = era5_pres['w'].sel(pressure_level=700)
        era5_data['rh_700'] = era5_pres['r'].sel(pressure_level=700)
        era5_data['q_700'] = era5_pres['q'].sel(pressure_level=700)
        era5_data['rh_1000'] = era5_pres['r'].sel(pressure_level=1000)
        era5_data['speed'] = np.hypot(era5_data['u10'], era5_data['v10'])
        era5_data['pv_700'] = era5_pres['pv'].sel(pressure_level=700)
        era5_data['pv_500'] = era5_pres['pv'].sel(pressure_level=500)
        
        era5_data = share.crop_era5(era5_data, rename=True, 
                                    domain=share.pac_domain)
        era5_flux = share.crop_era5(era5_flux, rename=True, 
                                    domain=share.pac_domain)
        # Calculate cold advection with wrapping and lat/lon names
        era5_data['cold_adv'] = cold_adv_periodic(era5_data)
        era5_data['cold_adv_smooth'] = share.smooth_data(era5_data['cold_adv'], 3)
        # Mean year
        climatology = era5_data.groupby('time.month').mean(dim='time')
        clim_rate = era5_flux.groupby('time.month').mean(dim='time')
        # deas
        years = era5_data.time.dt.year
        months = era5_data.time.dt.month
        num = len(years)
        # Anomaly time series
        for n, (year, month) in enumerate(zip(years, months)):
            share.progress_bar(n, num, f'Deseasonalizing ERA5...{int(year)}-{int(month)}')     
            era5_data[{"time": n }] -= climatology.sel(month=month)
        # deas
        years = era5_flux.time.dt.year
        months = era5_flux.time.dt.month
        num = len(years)
        # Anomaly time series
        for n, (year, month) in enumerate(zip(years, months)):
            share.progress_bar(n, num, f'Deseasonalizing ERA5 fluxess...{int(year)}-{int(month)}')     
            era5_flux[{"time": n }] -= clim_rate.sel(month=month)
        
        # fix a date issue?
        era5_flux['time'] = era5_data.time
        print('\nSaving to file...\n')
        era5_data.to_netcdf(file_era5)  
        era5_flux.to_netcdf(file_flux)
    
    # era5_flux['time'] = era5_data.drop_vars('number').time
    _, pc_enso = share.calc_eof(era5_data, 'sst', n_pc=4,
                                plot=False, region='equator', detrend=True)
    # Just so +ve PC1 is +ve ENSO, as per cloud_corr
    # pc_enso['PC1'] *= -1
    pc_enso[['PC1', 'PC2']] *= -1
    pc_enso = share.rotate_enso_eof(pc_enso)
    
    # Correlations from before
    corr = share.calc_corr_vect(era5_data, 'eis', pc_enso, 'PC2')
    share.plot_corr(corr, cbar_lab='R', lims=share.pac_domain,
              title='Correlation Between EIS and PC2 Mode')
    
    _, pc_700 = share.calc_eof(era5_data, var='theta_700', n_pc=1, plot=False,
                                region='tropics', detrend=True)
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = share.calc_corr_vect(era5_data, 'theta_700', pc_700, 'PC1')
    share.plot_corr(corr, cbar_lab='R', lims=share.pac_domain,
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1')
    
    # Single variable trajectories
    lcc_anom = share.isolate_ep_era5(era5_data, domain=share.ep_domain_360, 
                                     var='lcc')
    theta_anom = era5_data['theta_700'].sel(lat=slice(30, -30)).mean(dim=['lat',
                                                                          'lon'])

    # Nino 3.4 index and ONI for our data
    nino_data = calc_oni(era5_data)
    # Noaa-style detrending for clouds; include other variables we want 
    # which makes analysis a little easier
    nino_data[['PC1', 'C', 'E', 'simp_time', 'PC2']] = pc_enso[['PC1', 'C', 'E',
                                                                'simp_time',
                                                                'PC2']]
    nino_data['lcc_detr'] = 100 * polynomial_detrend(lcc_anom, nino_data['simp_time'],
                                                     6) 
    nino_data['lcc_clean'] = share.butter_lowpass_filter(nino_data['lcc_detr'], 1/8, 
                                                   1, 4)
    # to make +ve eof = el nino
    nino_data['theta_anom'] = signal.detrend(theta_anom)
    nino_data['PC_theta'] = pc_700['PC1']
    ## assign enso state
    nino_data = is_enso_monthly(nino_data)
          
if __name__ == "__main__":
    main()
