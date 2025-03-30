# -*- coding: utf-8 -*-
"""
Analyze radiation anomalies (LW, SW, Net, TOA, SFC) in the eastern pacific
and  observe correlations to ENSO phases
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import os
import pandas as pd
from scipy.signal import detrend
from scipy.stats import linregress


os.chdir('C:/Users/aakas/Documents/ENSO-Clouds/')
# We need to do the same thing as with the CERES data where annual
# composites are generated and used to construct El Nino/La Nina anomaly fields
import obs_scripts.shared_funcs as share

# Thermocline depth parameter
cline_depth = 100


def rad_trajectory(ceres, oni_idx, year, anom_rad, era5_ep, anom_sing):
    """
    Plots the "radiative trajectory", i.e. the net sfc and toa radiative
    effects over the course of an ENSO cycle (October-March)
    
    Assuming a mixed-layer depth of 50m, calculates the implied
    temperature change in the ocean (using surface radiation)
    
    Anom rad (dictionary with ENSO composites) 
    tells us behavior during a mean ENSO
    
    Era5_ep allows us to plot the actual temperature change during this period
    anom_sing achieves the same as anom_rad for this purpose
    """
    trajectory = ceres.sel(time=slice(f'{year}-9-01', f'{year+1}-04-30'))
    ocean = era5_ep.sel(time=slice(f'{year}-9-01', f'{year+1}-04-30'))['sst']
    ocean = np.array(ocean.mean(dim=['lat', 'lon']))
    ocean -= ocean[0]
    trajectory = trajectory.mean(dim=['lat', 'lon'])
    
    state = share.is_enso_oni(oni_idx, f'{year}.12')
    rad_state = anom_rad[state.replace(' ', '_').lower()]
    rel_traj = rad_state.sel(month=[9, 10, 11, 12, 1, 2, 3, 4])
    rel_traj = rel_traj.mean(dim=['lat', 'lon'])
    # Mean changes in SST
    sst_state = anom_sing[state.replace(' ', '_').lower()]
    sst_state = sst_state.sel(month=[9, 10, 11, 12, 1, 2, 3, 4])
    sst_state = np.array(sst_state['sst'].mean(dim=['lat', 'lon']))
    sst_state -= sst_state[0]
    
    # Get TOA/SFC components now
    traj_toa = trajectory['toa_net_all_mon']
    traj_sfc = trajectory['sfc_net_tot_all_mon']
    ocean_t = (traj_sfc * 3600 * 24 * 30) / (1000 * 4284 * cline_depth)
    total_oc_t = [np.sum(ocean_t[:x]) for x in range(1, 9)]    
    
    mean_toa = rel_traj['toa_net_all_mon']
    mean_sfc = rel_traj['sfc_net_tot_all_mon']
    mean_t = (mean_sfc * 3600 * 24 * 30) / (1000 * 4284 * cline_depth)
    mean_oc_t = [np.sum(mean_t[:x]) for x in range(1, 9)]    
    
    # Plots!
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('LCC Anomaly Impacts in EP 30°S-0°N, 120°W-80°W')
    plt.tight_layout()
    
    axs[0].plot(trajectory.time, traj_toa, label=f'{year} TOA', color='darkblue')
    axs[0].plot(trajectory.time, traj_sfc, label=f'{year} SFC', color='darkred')
    axs[0].plot(trajectory.time, mean_toa, label=f'Mean {state} TOA',
                alpha=0.7, linestyle='dashed', color='blue')
    axs[0].plot(trajectory.time, mean_sfc, label=f'Mean {state} SFC',
                alpha=0.7, linestyle='dashed', color='red')
    axs[0].set_title('Radiative Impacts')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Radiative anomaly (W/m2)')
    
    #axs[1].plot(trajectory.time, ocean_t, label=f'{year} Ocean Temp. Change',
    #            color='darkgreen')
    axs[1].plot(trajectory.time, total_oc_t, label=f'{year} Implied Temp. Change',
                color='indigo')
    #axs[1].plot(trajectory.time, mean_t, label=f'Mean {state} Change',
    #            linestyle='dashed', alpha=0.7, color='green')
    axs[1].plot(trajectory.time, mean_oc_t, label=f'Mean {state} Implied Change',
                linestyle='dashed', alpha=0.7, color='violet')
    axs[1].plot(trajectory.time, ocean, label='Real Ocean Change',
                color='lightcoral')
    axs[1].plot(trajectory.time, sst_state, alpha=0.7, linestyle='dashdot',
                color='crimson', label=f'Mean {state} Change')
    axs[1].legend()
    axs[1].grid() 
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Temp Anomaly (K)')
    axs[1].yaxis.tick_right()
    axs[1].set_title('Ocean Impacts')
    

def rad_type(ceres, oni_idx, year, anom_rad):
    """
    Similar to rad trajectory, but shows partitioning of energy into
    longwave and shortwave, and maybe clearsky/cloudy
    """
    trajectory = ceres.sel(time=slice(f'{year}-9-01', f'{year+1}-04-30'))
    trajectory = trajectory.mean(dim=['lat', 'lon'])
    
    state = share.is_enso_oni(oni_idx, f'{year}.12')
    rad_state = anom_rad[state.replace(' ', '_').lower()]
    rel_traj = rad_state.sel(month=[9, 10, 11, 12, 1, 2, 3, 4])
    rel_traj = rel_traj.mean(dim=['lat', 'lon'])
    
    # Get TOA/SFC components now
    traj_toa_sw = trajectory['toa_sw_all_mon']
    traj_toa_lw = trajectory['toa_lw_all_mon']
    traj_sfc_sw = trajectory['sfc_net_sw_all_mon']  
    traj_sfc_lw = trajectory['sfc_net_lw_all_mon']
    
    mean_toa_sw = rel_traj['toa_sw_all_mon']
    mean_toa_lw = rel_traj['toa_lw_all_mon']
    mean_sfc_sw = rel_traj['sfc_net_sw_all_mon']  
    mean_sfc_lw = rel_traj['sfc_net_lw_all_mon'] 
    
    # Subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle('Partition of Radiative Flux')
    plt.tight_layout()
    
    axs[0].plot(trajectory.time, traj_toa_sw, label=f'{year} SW',
                color='darkred')
    axs[0].plot(trajectory.time, traj_toa_lw, label=f'{year} LW',
                color='darkblue')
    axs[0].plot(trajectory.time, mean_toa_sw, label=f'Mean {state} SW',
                color='red', linestyle='dashed', alpha=0.7)
    axs[0].plot(trajectory.time, mean_toa_lw, label=f'Mean {state} LW',
                color='blue', linestyle='dashed', alpha=0.7)
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title('Top of Atmosphere')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Radiation Flux (W/m2)')
    
    axs[1].plot(trajectory.time, traj_sfc_sw, label=f'{year} SW',
                color='darkred')
    axs[1].plot(trajectory.time, traj_sfc_lw, label=f'{year} LW',
                color='darkblue')
    axs[1].plot(trajectory.time, mean_sfc_sw, label=f'Mean {state} SW',
                color='red', linestyle='dashed', alpha=0.7)
    axs[1].plot(trajectory.time, mean_sfc_lw, label=f'Mean {state} LW',
                color='blue', linestyle='dashed', alpha=0.7)
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title('Surface')
    axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('Radiation Flux (W/m2)')


def ep_cp_ensos_dec(pc_enso):
    """
    Uses the E and C indices to define E vs C dominant nature
    of ENSO events
    
    Uses E, C state in december
    
    Returns a df containing year, enso state, fe and fc, calculated as
    fe = E / (E + C) and complementary
    """
    # Offsets to ensure we are comparing the right years
    e_class = pc_enso.query('month==12').reset_index()
    c_class = pc_enso.query('month==12').reset_index()
    
    fe = e_class['E'] / (np.abs(c_class['C']) + np.abs(e_class['E']))
    fc = c_class['C'] / (np.abs(c_class['C']) + np.abs(e_class['E']))
    
    pattern = pd.DataFrame({'year': e_class.year,
                            'fe': fe, 'fc': fc})
    
    return pattern


def ep_rad_impacts(ceres, oni_idx):
    """
    Tabluates the radiative impact of enso during every year, expressed
    as the cumulative temp change from radiation in the EP
    """
    # Remove 2023 for incompplete data (fix this now?)
    years = np.unique(ceres.time.dt.year)[:-1]
    max_change = np.zeros(years.shape)
    
    for n, year in enumerate(years):
        trajectory = ceres.sel(time=slice(f'{year}-9-01', f'{year+1}-04-30'))
        trajectory = trajectory.mean(dim=['lat', 'lon'])
        
        traj_sfc = trajectory['sfc_net_tot_all_mon']
        ocean_t = (traj_sfc * 3600 * 24 * 30) / (1000 * 4284 * cline_depth)
        max_change[n] = np.sum(ocean_t)
        
    impact = pd.DataFrame({'year': years, 'temp': max_change,
                           'state': ''})
    
    for n, year in enumerate(impact.year):
        impact.state[n] = share.is_enso_oni(oni_idx, f'{year}.12')
        
    return impact  


def plot_cooling_cont(impact):
    """
    Subplot of temperature impact in the eastern pacific vs. 
    fe or fc contribution, along with best fit line
    """
    # Create two linregress objects
    result_c = linregress(impact['fc'], impact['temp'])
    result_e = linregress(impact['fe'], impact['temp'])
    # Subplots
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
    fig.suptitle('ENSO Pattern Impacts in Eastern Pacific')
    plt.tight_layout()
    # Setup for best fits
    c_range = np.linspace(min(impact['fc']), max(impact['fc']))
    e_range = np.linspace(min(impact['fe']), max(impact['fe']))
    # Plot- label scatter by event class
    colors = ['darkred', 'darkblue', 'violet']
    for n, state in enumerate(['El Nino', 'La Nina', 'Neutral']):
        df = impact.query(f'state == "{state}"')
        axs[0].scatter(df['fc'], df['temp'], label=state,
                       c=colors[n])
        axs[1].scatter(df['fe'], df['temp'], label=state,
                       c=colors[n])
    # Best fit lines
    axs[0].plot(c_range, result_c.slope * c_range + result_c.intercept,
                color='red', linestyle='dashed', label=f'R²={result_c.rvalue**2:.2f}' +\
                    f'\np = {result_c.pvalue:.3f} << 0.01')
    axs[0].grid()
    axs[0].set_xlabel('Frac. C Index')
    axs[0].set_ylabel('EP Temp Change (K)')
    axs[0].legend()
    axs[0].set_title('CP Variability')
    
    
    axs[1].plot(e_range, result_e.slope * e_range + result_e.intercept,
                color='red', linestyle='dashed', label=f'R²={result_e.rvalue**2:.2f}' +\
                    f'\np = {result_e.pvalue:.3f} > 0.01')
    axs[1].grid()
    axs[1].set_xlabel('Frac. E Index')
    # axs[1].set_ylabel('EP Temp Change (K)')
    axs[1].legend()
    axs[1].set_title('EP Variability')


def plot_cloud_cover(era5_ep, syn_ep, year):
    """
    Similar to the radiative trajectories, plots the cloud cover changes
    from ERA5 and ceres syn; mean and individual year's trajectory
    """
    cloud = era5_ep.sel(time=slice(f'{year}-9-01',
                                   f'{year+1}-04-30')).lcc.mean(dim=['lat', 
                                                                     'lon'])
    syn_cloud = syn_ep.sel(time=slice(f'{year}-9-01', 
                                      f'{year+1}-04-30'))['cldarea_low_mon'].mean(dim=['lat', 
                                                                                       'lon'])
    plt.plot(share.time, 100*share.data, label='ERA5')
    plt.plot(syn_cloud.time, syn_cloud.data, label='CERES SYN'); plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Low Cloud Cover Anomaly')
    
    
def seasonal_cc_change(era5_ep, syn_ep):
    """
    Returns a pandas df containing the mean cloud cover anomaly between ocrober
    and february, along with the maximum anomaly in that period, and the year
    when it happened
    
    Also ensures both are in % units
    """
    years = np.unique(syn_ep.time.dt.year)[:-1]
    
    low_clouds = pd.DataFrame({'year': years,
                               'era5_mean_lcc': np.zeros(years.shape),
                               'era5_max_lcc': np.max(years.shape),
                               'syn_mean_lcc': np.zeros(years.shape),
                               'syn_max_lcc': np.zeros(years.shape)})
    
    for n, year in enumerate(years):
        era5 = np.asarray(era5_ep.sel(time=slice(f'{year}-10-01',
                                      f'{year+1}-03-30')).lcc.mean(dim=['lat', 
                                                                        'lon']))
        syn_cloud = np.asarray(syn_ep.sel(time=slice(f'{year}-10-01',
                                          f'{year+1}-03-30'))['cldarea_low_mon'].mean(dim=['lat', 'lon']))
        
        low_clouds['era5_mean_lcc'][n] = 100 * np.mean(era5)
        low_clouds['era5_max_lcc'][n] = 100 * era5[np.argmax(np.abs(era5))]
        low_clouds['syn_mean_lcc'][n] = np.mean(syn_cloud)
        low_clouds['syn_max_lcc'][n] = syn_cloud[np.argmax(np.abs(syn_cloud))]
    
    return low_clouds       


def main():
    global lcc_syn
    file_ceres = 'era5_reanal/timeseries/ceres_ebaf.nc'
    dir_anom = 'era5_reanal/anomalies/'  
    file_era5 = 'era5_reanal/timeseries/era5_anom.nc'
    file_syn = 'era5_reanal/timeseries/ceres_syn.nc'
    
    domain = share.eqp_domain_360 
    
    lat_bounds = domain[:2]
    lon_bounds = domain[2:]
    
    files = [file_era5, file_ceres, dir_anom, file_syn]
    if all([os.path.exists(file) for file in files]):
        ceres = xr.open_dataset(file_ceres)
        ceres_ep = ceres.sel(lat=slice(*lat_bounds), lon=slice(*lon_bounds))
        syn_ep = xr.open_dataset(file_syn).sel(lat=slice(*lat_bounds), 
                                               lon=slice(*lon_bounds))
        
        era5_ep = xr.open_dataset(file_era5)
        era5_ep['lon'] = (era5_ep.lon + 360) % 360
        era5_ep = era5_ep.sel(lat=slice(*lat_bounds[::-1]), 
                              lon=slice(*lon_bounds))
        
        anom_pres, anom_ep, anom_rad = share.load_composites(dir_anom, clim=False)
        # Remember, thesee are not cropped yet
        for state, value in anom_rad.items():
            anom_rad[state] = value.sel(lat=slice(*lat_bounds), 
                                        lon=slice(*lon_bounds))
        for state, value in anom_ep.items():
            value['lon'] = (value.lon + 360) % 360
            anom_ep[state] = value.sel(lat=slice(*lat_bounds[::-1]), 
                                         lon=slice(*lon_bounds))   
    else:
        print('ERROR: Run cloud_corr.py to generate ceres timeseries, or' +
              'divergence.py for mean El Nino/La Nina years')
        return
    
    # Obtain PCs and Nino index
    pc_enso = pd.read_csv('misc_data/enso_pcs.csv')
    # Contains nino 3.4 anomaly
    nino_idx = share.load_nino_idx('misc_data/nino_all.csv')
    nino_idx['3.4_anom'] = detrend(nino_idx['3.4_anom'])
    # Create a lagged index by arbitrary number of months
    nino_idx['3.4_anom_lag'] = nino_idx['3.4_anom'].shift(2)
    # ONI Index
    oni_idx = share.load_oni_idx('misc_data/oni_index.txt')
    
    # Integrated trajectories through time
    # rad_trajectory(ceres_ep, oni_idx, 2020, anom_rad, era5_ep, anom_ep)
    # Partition into longwave/shortwave
    rad_type(ceres_ep, oni_idx, 2020, anom_rad)
    
    # Analyze quantitatively the cumulative impact based on ENSO pattern
    # (didnt pan out)
    # ep_cp_idx = ep_cp_ensos(pc_enso)
    # impact = ep_rad_impacts(ceres_ep, oni_idx)
    # Left join of data
    # impact = impact.merge(ep_cp_idx, how='left')    
    # plot_cooling_cont(impact)
    
    # We can check the era5-like lag relationships with C/E
    lcc_syn = syn_ep['cldarea_low_mon'].mean(dim=['lat', 'lon'])
    share.plot_combined(pc_enso['C'][12:], lcc_syn, syn_ep.time, 
                        'C', 'syn lcc')
    # Roughly in-phase, no relationship with E/PC1
    lcc_era5 = era5_ep['lcc'].mean(dim=['lat', 'lon'])
    share.plot_combined(pc_enso['C'], lcc_era5, era5_ep.time, 
                        'C', 'era5 lcc')
    # lag is far more distinnct!
    # Check PC2 !!!
    lcc_syn_sm = share.butter_lowpass_filter(lcc_syn, 1/6, 1)
    
    share.plot_combined(pc_enso['PC2'][12:], lcc_syn, syn_ep.time, 
                        'PC2', 'syn lcc')
    share.plot_combined(pc_enso['PC2'][12:], lcc_syn_sm, syn_ep.time, 
                        'PC2', 'syn lcc lowpass')
    
if __name__ == '__main__':
    main()
