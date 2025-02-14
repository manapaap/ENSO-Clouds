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
import scipy.signal as signal
from scipy.optimize import curve_fit
import scipy.stats as stats
import warnings
import pandas as pd


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
from obs_scripts.vis_clouds import progress_bar
from obs_scripts.divergence import crop_era5
import obs_scripts.cloud_corr as cloud
from obs_scripts.ceres_ep import plot_regression


# Some domains for convenience
cz_domain = [-30, 30, 120, -80]
# For plotting correlations and scalar
ep_region = [-20, 0, 240, 280] # Can make an argument for [-20, 0]
# Larger pacific
pac_domain = [-50, 50, 120, -60]
# NIno 3.4
nino_domain = [-5, 5, -170, -120]


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


def isolate_ep(era5_data, ep_domain=ep_region, var='lcc'):
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
    return era5_ep.mean(dim=['lat', 'lon'])


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = signal.butter(order, normalCutoff, btype='low', analog=False)  # Use analog=False for a digital filter
    return b, a


def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = signal.filtfilt(b, a, data)  # Use filtfilt for zero-phase filtering
    return y


def red_power(f, autocorr, A):
    """
    Function for red noise spectrum. Will be fit to PSD. 
    """ 
    rs = A * (1.0 - autocorr**2) /(1. -(2.0 * autocorr * np.cos(f *2.0 * np.pi)) + autocorr**2)
    return rs


def plot_psd(array, nperseg=256, sig=0.99, cutoff=1, 
             period=1, nfft=512, name='SST'):
    """
    Plots the psd and red noise null hypothesis to check for significant peaks
    under "cutoff"
    
    Returns relevant parameters to reconstruct the figure
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
    red_params, red_covar = curve_fit(red_power, f, Pxx, p0=(corr_1, 1))
    red_fitted = red_power(f, *red_params)
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
        
    plt.semilogx(f, Pxx, label=f'{name}, Not Significant')
    plt.vlines([(12*3)**-1, (12*7)**-1], min(Pxx), max(Pxx),
                  label='ENSO Freqs', color='red', linestyle='dashed')
    plt.plot(f, red_fitted, label='Fitted Red Noise')
    plt.plot(f, f_stat * red_fitted, label=f'{sig} Significance')
    plt.grid()
    plt.ylabel('Power')
    plt.xlabel('Frequency (Cycles / month)')
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


def calc_oni(era5_data):
    """
    Calculates the oceanic Nino index for our data from 1940-present
    """
    nino_data = era5_data['sst'].sel(lat=slice(*nino_domain[:2][::-1]), 
                                     lon=slice(*nino_domain[2:]))
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
    
    
def main():    
    global era5_data, pc_enso, lcc_anom, nino_data
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
        era5_data['t_500'] = era5_pres['t'].sel(pressure_level=500)
        era5_data['rh_700'] = era5_pres['r'].sel(pressure_level=700)
        era5_data['rh_1000'] = era5_pres['r'].sel(pressure_level=1000)
        era5_data['speed'] = np.hypot(era5_data['u10'], era5_data['v10'])
        era5_data['eis'], era5_data['lts'], era5_data['theta_700'] = cloud.calc_eis(era5_pres,
                                                                      truncate=False)
        
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
    
    _, pc_enso = cloud.calc_eof(era5_data, 'sst', n_pc=2,
                                plot=False, region='equator', detrend=True)
    # Just so +ve PC1 is +ve ENSO, as per cloud_corr
    # pc_enso['PC1'] *= -1
    pc_enso = cloud.rotate_enso_eof(pc_enso)
    
    # Correlations from before
    corr = cloud.calc_corr_vect(era5_data, 'lcc', pc_enso, 'C')
    cloud.plot_corr(corr, cbar_lab='R', lims=pac_domain,
              title='Correlation Between LCC and C Mode')
    
    _, pc_700 = cloud.calc_eof(era5_data, var='theta_700', n_pc=1, plot=False,
                                region='tropics', detrend=True)
    # The plots look very similar to Nino 3.4 correlation, so we ignore this
    corr = cloud.calc_corr_vect(era5_data, 'theta_700', pc_700, 'PC1')
    cloud.plot_corr(corr, cbar_lab='R', lims=pac_domain,
              title='Correlation Between Θ₇₀₀ Anom and Θ₇₀₀ PC1')
    
    # Single variable trajectories
    lcc_anom = isolate_ep(era5_data, ep_region, 'lcc')
    theta_anom = era5_data['theta_700'].sel(lat=slice(30, -30)).mean(dim=['lat',
                                                                          'lon'])

    # Nino 3.4 index and ONI for our data
    nino_data = calc_oni(era5_data)
    # Noaa-style detrending for clouds; include other variables we want 
    # which makes analysis a little easier
    nino_data['lcc_detr'] = 100 * noaa_detrend(lcc_anom, 10) 
    nino_data[['PC1', 'C', 'E']] = pc_enso[['PC1', 'C', 'E']]
    # to make +ve eof = el nino
    nino_data[['PC1', 'C', 'E']] *= -1
    nino_data['theta_anom'] = signal.detrend(theta_anom)
    nino_data['PC_theta'] = pc_700['PC1']
    ## assign enso state
    nino_data = is_enso_monthly(nino_data)
          
if __name__ == "__main__":
    main()
